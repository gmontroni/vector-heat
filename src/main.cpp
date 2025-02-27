#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include "polyscope/point_cloud.h"
#include "geometrycentral/pointcloud/point_position_geometry.h"
#include "geometrycentral/pointcloud/point_cloud_heat_solver.h"
#include "geometrycentral/pointcloud/point_cloud_io.h"

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

// #include <iostream>
#include <fstream>
#include <cstdlib> // getenv
#include "args/args.hxx"
#include "imgui.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;
using namespace geometrycentral::pointcloud;

// == Geometry-central data (Meshes)
std::unique_ptr<ManifoldSurfaceMesh> mesh;
std::unique_ptr<VertexPositionGeometry> geometry;

// == Geometry-central data (Points Cloud)
std::unique_ptr<PointCloud> cloud;
std::unique_ptr<PointPositionGeometry> point_geometry;

// Polyscope visualization handle, to quickly add data to the surface
polyscope::SurfaceMesh *psMesh;

int main(int argc, char **argv) {

  // Configure the argument parser
  args::ArgumentParser parser("geometry-central & Polyscope example project");
  args::Positional<std::string> inputFilename(parser, "mesh", "A mesh file.");

  // Parse args
  try {
    parser.ParseCLI(argc, argv);
  } catch (args::Help &h) {
    std::cout << parser;
    return 0;
  } catch (args::ParseError &e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;
  }

  // Make sure a mesh name was given
  if (!inputFilename) {
    std::cerr << "Please specify a mesh file as argument" << std::endl;
    return EXIT_FAILURE;
  }

  // Initialize polyscope
  polyscope::init();

  // Load mesh
  std::tie(mesh, geometry) = readManifoldSurfaceMesh(args::get(inputFilename));

  // Load point cloud
  std::tie(cloud, point_geometry) = readPointCloud(args::get(inputFilename));

  // Register the mesh with polyscope
  psMesh = polyscope::registerSurfaceMesh(
      polyscope::guessNiceNameFromPath(args::get(inputFilename)),
      geometry->inputVertexPositions, mesh->getFaceVertexList(),
      polyscopePermutations(*mesh));

  // Set normals
  point_geometry->requireNormals();
  PointData<Vector3> vNormals(*cloud);
  for (Point p : cloud->points()) {
    vNormals[p] = point_geometry->normals[p];
    // std::cout << "normal for point " << p << " is " << point_geometry->normals[p] << "\n";
  }

  // Set tangent basis
  point_geometry->requireTangentBasis();
  PointData<Vector3> vBasisX(*cloud);
  PointData<Vector3> vBasisY(*cloud);
  for (Point p : cloud->points()) {
    vBasisX[p] = point_geometry->tangentBasis[p][0];
    vBasisY[p] = point_geometry->tangentBasis[p][1];
    // std::cout << "basis x for point " << p << " is " << point_geometry->tangentBasis[p][0] << "\n";
  }

  // Create the solver
  double tCoef = 1.0;
  PointCloudHeatSolver solver(*cloud, *point_geometry, tCoef);

  // // Compute a timescale
  // double meanEdgeLength = 0.;
  // double shortTime = 0.0;
  // for (surface::Edge e : point_geometry->tuftedMesh->edges()) {
  //   meanEdgeLength += point_geometry->tuftedGeom->edgeLengths[e];
  // }
  // meanEdgeLength /= point_geometry->tuftedMesh->nEdges();
  // shortTime = tCoef * meanEdgeLength * meanEdgeLength;

  // Set laplacian
  point_geometry->requireLaplacian();
  Eigen::SparseMatrix<double> L = point_geometry->laplacian;

  // Set connection laplacian
  point_geometry->requireConnectionLaplacian();
  Eigen::SparseMatrix<double> cL = point_geometry->connectionLaplacian;

  // Set mass matrix
  point_geometry->tuftedGeom->requireVertexLumpedMassMatrix();
  Eigen::SparseMatrix<double> massMat = point_geometry->tuftedGeom->vertexLumpedMassMatrix; // Mass matrix

  Eigen::SparseMatrix<double> massMatcomplex = complexToReal(massMat.cast<std::complex<double>>().eval());

  // Set operator
  // Eigen::SparseMatrix<double> vectorOp = massMatcomplex + shortTime * cL; // vectorOp = operador M - tL
  // vectorHeatSolver.reset(new PositiveDefiniteSolver<double>(vectorOp));

  // Pick a source point
  Point pSource = cloud->point(7);

  // Compute parallel transport
  Vector2 sourceVec{1, 1};
  PointData<Vector2> transport = solver.transportTangentVector(pSource, sourceVec);
 
  // get home directory
  std::string homeDir = getenv("HOME");
  std::string filePath = homeDir + "/Documents/Parallel Transport/verification";
  std::string normalPath = filePath + "/normal.txt";
  std::string lPath = filePath + "/laplacian_matrix.txt";
  std::string cLPath = filePath + "/connection_laplacian_matrix.txt";

  // save basis X to file
  std::ofstream basisXFile(filePath + "/basisX.txt");
  if (basisXFile.is_open()) {
      for (Point p : cloud->points()) {
          Vector3 basisX = vBasisX[p];
          basisXFile << basisX.x << " " << basisX.y << " " << basisX.z << "\n";
      }
      basisXFile.close();
  } else {
      std::cerr << "Unable to open file for writing" << std::endl;
  }

  // save basis Y to file
  std::ofstream basisYFile(filePath + "/basisY.txt");
  if (basisYFile.is_open()) {
      for (Point p : cloud->points()) {
          Vector3 basisY = vBasisY[p];
          basisYFile << basisY.x << " " << basisY.y << " " << basisY.z << "\n";
      }
      basisYFile.close();
  } else {
      std::cerr << "Unable to open file for writing" << std::endl;
  }

  // save normals to file
  std::ofstream normalFile(normalPath);
  if (normalFile.is_open()) {
      for (Point p : cloud->points()) {
          Vector3 normal = vNormals[p];
          normalFile << normal.x << " " << normal.y << " " << normal.z << "\n";
      }
      normalFile.close();
  } else {
      std::cerr << "Unable to open file for writing" << std::endl;
  }

  // save laplacian matrix to file
  std::ofstream outFile(lPath);
  if (outFile.is_open()) {
      for (int k = 0; k < L.outerSize(); ++k) {
          for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
              outFile << it.row() << " " << it.col() << " " << it.value() << "\n";
          }
      }
      outFile.close();
  } else {
      std::cerr << "Unable to open file for writing" << std::endl;
  }

  // save connection laplacian matrix to file
  std::ofstream cLOutFile(cLPath);
  if (cLOutFile.is_open()) {
      for (int k = 0; k < cL.outerSize(); ++k) {
          for (Eigen::SparseMatrix<double>::InnerIterator it(cL, k); it; ++it) {
              cLOutFile << it.row() << " " << it.col() << " " << it.value() << "\n";
          }
      }
      cLOutFile.close();
  } else {
      std::cerr << "Unable to open file for writing" << std::endl;
  }

  // save mass matrix to file
  std::ofstream massMatFile(filePath + "/mass_matrix.txt");
  if (massMatFile.is_open()) {
      for (int k = 0; k < massMat.outerSize(); ++k) {
          for (Eigen::SparseMatrix<double>::InnerIterator it(massMat, k); it; ++it) {
              massMatFile << it.row() << " " << it.col() << " " << it.value() << "\n";
          }
      }
      massMatFile.close();
  } else {
      std::cerr << "Unable to open file for writing" << std::endl;
  }

  // save mass matrix complex to file
  std::ofstream massMatComplexFile(filePath + "/mass_matrix_complex.txt");
  if (massMatComplexFile.is_open()) {
      for (int k = 0; k < massMatcomplex.outerSize(); ++k) {
          for (Eigen::SparseMatrix<double>::InnerIterator it(massMatcomplex, k); it; ++it) {
              massMatComplexFile << it.row() << " " << it.col() << " " << it.value() << "\n";
          }
      }
      massMatComplexFile.close();
  } else {
      std::cerr << "Unable to open file for writing" << std::endl;
  }

  psMesh->addVertexTangentVectorQuantity("Parallel Transport", transport, vBasisX, vBasisY);
  psMesh->addVertexVectorQuantity("Normal", vNormals);
  psMesh->addVertexVectorQuantity("Basis X", vBasisX);
  psMesh->addVertexVectorQuantity("Basis Y", vBasisY);

  // Give control to the polyscope gui
  polyscope::show();

  return EXIT_SUCCESS;
}
