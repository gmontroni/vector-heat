#include "geometrycentral/surface/meshio.h"
// #include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/surface/direction_fields.h"

#include "geometrycentral/pointcloud/point_cloud_io.h"
#include "polyscope/point_cloud.h"
#include "geometrycentral/pointcloud/point_position_geometry.h"
#include "geometrycentral/pointcloud/point_cloud_heat_solver.h"

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

  // Set parallel transport oriented (Fazer)

  // Set laplacian
  point_geometry->requireLaplacian();
  Eigen::SparseMatrix<double> L = point_geometry->laplacian;

  // Set connection laplacian
  point_geometry->requireConnectionLaplacian();
  Eigen::SparseMatrix<double> cL = point_geometry->connectionLaplacian;

  // Set direction field
  auto dField = geometrycentral::surface::computeSmoothestVertexDirectionField(*geometry);

//   // Copy the result to a VertexData vector
//   geometry->requireVertexIndices();
//   VertexData<Vector2> toReturn(*mesh);
//   for (Vertex v : mesh->vertices()) {
//     toReturn[v] = Vector2::fromComplex(cL(geometry->vertexIndices[v]));
//     toReturn[v] = unit(toReturn[v]);
//   }
 
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

  // Set vertex tangent spaces (Mesh)
  geometry->requireVertexTangentBasis();
  VertexData<Vector3> vmBasisX(*mesh);
  VertexData<Vector3> vmBasisY(*mesh);
  for (Vertex v : mesh->vertices()) {
    vmBasisX[v] = geometry->vertexTangentBasis[v][0];
    vmBasisY[v] = geometry->vertexTangentBasis[v][1];
  }

     auto vField =
            geometrycentral::pointcloud::PointCloudHeatSolver::PointCloudHeatSolver(cloud*, point_geometry*)

//   auto vField =
//       geometrycentral::surface::computeVertexConnectionLaplacian(*geometry);
//   auto vField =
//          geometrycentral::surface::computeSmoothestVertexDirectionField(*geometry);
//   psMesh->addVertexTangentVectorQuantity("VF", vField, vBasisX, vBasisY);
  psMesh->addVertexTangentVectorQuantity("Direction Field", dField, vmBasisX, vmBasisY);
  psMesh->addVertexVectorQuantity("Normal", vNormals);
  psMesh->addVertexVectorQuantity("Basis X", vBasisX);
  psMesh->addVertexVectorQuantity("Basis Y", vBasisY);

  // Give control to the polyscope gui
  polyscope::show();

  return EXIT_SUCCESS;
}
