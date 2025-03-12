#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/surface/vector_heat_method.h"
#include "polyscope/surface_mesh.h"
#include "geometrycentral/surface/direction_fields.h"

#include "polyscope/point_cloud.h"
#include "geometrycentral/pointcloud/point_position_geometry.h"
#include "geometrycentral/pointcloud/point_cloud_heat_solver.h"
#include "geometrycentral/pointcloud/point_cloud_io.h"

#include "polyscope/polyscope.h"

#include <iostream>
#include <typeinfo>
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

// == Geometry-central data (Tufted Mesh)
std::unique_ptr<SurfaceMesh> tuftedMesh;
std::unique_ptr<EdgeLengthGeometry> tuftedGeom;

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
  PointData<Vector3> pNormals(*cloud);
  for (Point p : cloud->points()) {
    pNormals[p] = point_geometry->normals[p];
    // std::cout << "normal for point " << p << " is " << point_geometry->normals[p] << "\n";
  }

  // Set tangent basis
  point_geometry->requireTangentBasis();
  PointData<Vector3> pBasisX(*cloud);
  PointData<Vector3> pBasisY(*cloud);
  for (Point p : cloud->points()) {
    pBasisX[p] = point_geometry->tangentBasis[p][0];
    pBasisY[p] = point_geometry->tangentBasis[p][1];
    // std::cout << "basis x for point " << p << " is " << point_geometry->tangentBasis[p][0] << "\n";
  }

  // Create the solver (Points Cloud)
  // double tCoef = 1.0;
  PointCloudHeatSolver pchSolver(*cloud, *point_geometry, 1.0);
  // PointCloudHeatSolver pchSolver(*cloud, *point_geometry); // default tCoef = 1.0

  // construct a solver (Mesh)
  VectorHeatMethodSolver vhmSolver(*geometry);

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
  
  // Compute parallel transport (Points Cloud) 
  Point pSource = cloud->point(600);
  Vector2 sourcePoint{-0.0683853626, 0.997658968};
  PointData<Vector2> pchTransport = pchSolver.transportTangentVector(pSource, sourcePoint);  // Only one source point

  // Smoothing test
  // PointCloudHeatSolver pchSolversm(*cloud, *point_geometry, 1.0);
  // PointCloudHeatSolver pchSolversmo(*cloud, *point_geometry, 0.001);

  // PointData<Vector2> pchTransportsm = pchSolversm.transportTangentVector(cloud->point(600), Vector2{-0.0683853626, 0.997658968});
  // PointData<Vector2> pchTransportsmo = pchSolversmo.transportTangentVector(cloud->point(600), Vector2{-0.0683853626, 0.997658968});

  // PointData<double> smoError(*cloud);
  // for (Point p : cloud->points()) {
  //     // Point p = cloud->point(p.getIndex());
  //     Vector2 cns = pchTransportsm[p];                      // Campo não suavizado
  //     Vector2 cs = pchTransportsmo[p];                      // Campo suavizado
  //     double dotProduct = cns.x * cs.x + cns.y * cs.y;      // Produto escalar
  //     smoError[p] = std::abs(1 - dotProduct);               // Erro
  // }
  //

  PointData<Vector2> initialVec(*cloud);
  initialVec[pSource] = sourcePoint;

  // // Compute parallel transport (Mesh)
  // Vertex vSource = mesh->vertex(0);
  // Vector2 sourceVecMesh{1, 2};
  // VertexData<Vector2> vhmTransport = vhmSolver.transportTangentVector(vSource, sourceVecMesh);  // Only one source vertex

  // Pick some source points
  std::vector<std::vector<Point>> idxs;
  idxs.push_back({cloud->point(600), cloud->point(1560), cloud->point(2520), cloud->point(3528), 
                  cloud->point(3000), cloud->point(2039), cloud->point(72), cloud->point(1032)});
  std::vector<std::vector<Vector2>> vecs2;
  vecs2.push_back({ Vector2{-0.0683853626, 0.997658968}, Vector2{-0.0989825577, 0.995089114}, Vector2{-0.0809355006, -0.996719301}, Vector2{-0.0456265137, -0.998958528},
                    Vector2{-0.0295587238, -0.999562979}, Vector2{-0.999998152, -0.0018937682}, Vector2{-0.999999583, -0.000849745411}, Vector2{-0.108760782, 0.994067907}});

  // create a vector of tuples with points and vectors
  std::vector<std::tuple<Point, Vector2>> sources;
  // create a for acessing for each element of idxs and vecs2
  for (size_t i = 0; i < idxs.size(); i++) {
      for (size_t j = 0; j < idxs[i].size(); j++) {
          sources.push_back(std::make_tuple(idxs[i][j], vecs2[i][j]));
          // std::cout << "idxs elements: " << idxs[i][j] << std::endl;
          // std::cout << "vecs2 elements: " << vecs2[i][j] << std::endl;
      }
  }

  // Compute parallel transport with sources points
  PointData<Vector2> pchTransportSourcesPoints = pchSolver.transportTangentVectors(sources);

  // Create a null PointData<Vector2> object
  PointData<Vector2> field(*cloud);
  for (size_t i = 0; i < idxs.size(); i++) {
    for (size_t j = 0; j < idxs[i].size(); j++) {
      field[idxs[i][j]] = vecs2[i][j];
    }
  }

  // Pick random source points and vectors
  std::vector<std::vector<Point>> idxsrand;
  idxsrand.push_back({cloud->point(0), cloud->point(1392), cloud->point(2881)});

  std::vector<std::vector<Vector2>> vecs2rand;
  vecs2rand.push_back({ Vector2{1, 2}, Vector2{3, 4}, Vector2{5, 6}});

  // create a vector of tuples with points and vectors
  std::vector<std::tuple<Point, Vector2>> sourcesrand;
  for (size_t i = 0; i < idxsrand.size(); i++) {
      for (size_t j = 0; j < idxsrand[i].size(); j++) {
          sourcesrand.push_back(std::make_tuple(idxsrand[i][j], vecs2rand[i][j]));
      }
  }

  // Compute parallel transport with Random Vectors
  PointData<Vector2> pchTransportRandSources = pchSolver.transportTangentVectors(sourcesrand);

  // Create a null PointData<Vector2> object
  PointData<Vector2> fieldrand(*cloud);
  for (size_t i = 0; i < idxsrand.size(); i++) {
    for (size_t j = 0; j < idxsrand[i].size(); j++) {
      fieldrand[idxsrand[i][j]] = vecs2rand[i][j];
    }
  }

  // Compute Direction Field (Mesh)
  VertexData<Vector2> directions = computeSmoothestVertexDirectionField(*geometry);

  PointData<double> error(*cloud);
  for (Point p : cloud->points()) {
      // Point p = cloud->point(p.getIndex());
      Vector2 cns = pchTransport[p];                    // Campo não suavizado
      Vector2 cs = pchTransportSourcesPoints[p];         // Campo suavizado
      double dotProduct = cns.x * cs.x + cns.y * cs.y;  // Produto escalar
      error[p] = std::abs(1 - dotProduct);               // Erro
  }

  // // Compute error abs(1 - dot product) of the directions elements VertexData<Vector2> with the parallel transport elements PointData<Vector2> of each point
  // VertexData<double> error(*mesh); // Inicialize o VertexData para armazenar o erro
  // for (Vertex v : mesh->vertices()) {
  //     Vector2 direction = directions[v];
  //     Point p = cloud->point(v.getIndex()); // Supondo que o índice do vértice corresponda ao ponto
  //     Vector2 transport = pchTransportSourcesPoints[p];
  //     double dotProduct = direction.x * transport.x + direction.y * transport.y;
  //     error[v] = std::abs(1 - dotProduct);
  // }

  // get home directory
  std::string homeDir = getenv("HOME");
  std::string filePath = homeDir + "/Documents/Parallel Transport/verification";
  std::string normalPath = filePath + "/normal.txt";
  std::string lPath = filePath + "/laplacian_matrix.txt";
  std::string cLPath = filePath + "/connection_laplacian_matrix.txt";

  // // read an initial field (X0) from file
  // std::string line;
  // std::ifstream
  // file(filePath + "/X0.txt");
  // std::vector<Vector2> X0;
  // if (file.is_open()) {
  //     while (std::getline(file, line)) {
  //         std::istringstream iss(line);
  //         double x, y;
  //         iss >> x >> y;
  //         X0.push_back(Vector2{x, y});
  //     }
  //     file.close();
  // } else {
  //     std::cerr << "Unable to open file for reading" << std::endl;
  // }

  // // create a vector of tuples with points and vectors
  // std::vector<std::tuple<Point, Vector2>> sources;
  // for (size_t i = 0; i < cloud->nPoints(); i++) {
  //     sources.push_back(std::make_tuple(cloud->point(i), X0[i]));
  // }

  // VertexData<Vector2> dField = computeSmoothestVertexDirectionField(*geometry);
  // std::vector<std::tuple<Point, Vector2>> sources;
  // for (size_t i = 0; i < cloud->nPoints(); i++) {
  //     sources.push_back(std::make_tuple(cloud->point(i), dField[i]));
  // }

  // PointData<Vector2> X = pchSolver.transportTangentVectors(sources);

  // save basis X to file
  std::ofstream basisXFile(filePath + "/basisX.txt");
  if (basisXFile.is_open()) {
      for (Point p : cloud->points()) {
          Vector3 basisX = pBasisX[p];
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
          Vector3 basisY = pBasisY[p];
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
          Vector3 normal = pNormals[p];
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

  // Tufted triangulation
  point_geometry->requireTuftedTriangulation();
  auto& tuftedMesh = point_geometry->tuftedMesh;
  auto& tuftedGeom = point_geometry->tuftedGeom;
  std::vector<Eigen::Vector3d> vertexPositions;
  for (Vertex v : tuftedMesh->vertices()) {
      auto pos = geometry->inputVertexPositions[v];
      vertexPositions.push_back(Eigen::Vector3d(pos.x, pos.y, pos.z));
  }
  std::vector<std::vector<size_t>> faceVertexList;
  for (Face f : tuftedMesh->faces()) {
      std::vector<size_t> face;
      for (Vertex v : f.adjacentVertices()) {
          face.push_back(v.getIndex());
      }
      faceVertexList.push_back(face);
  }

  // Set vertex tangent spaces
  geometry->requireVertexTangentBasis();
  VertexData<Vector3> vBasisX(*mesh);
  VertexData<Vector3> vBasisY(*mesh);
  for (Vertex v : mesh->vertices()) {
    vBasisX[v] = geometry->vertexTangentBasis[v][0];
    vBasisY[v] = geometry->vertexTangentBasis[v][1];
  }

  // Tufted triangulation plot
  polyscope::registerSurfaceMesh("Tufted Triangulation", vertexPositions, faceVertexList);
  
  // Add fields to the surface
  psMesh->addVertexTangentVectorQuantity("Direction Field", directions, vBasisX, vBasisY);
  psMesh->addVertexTangentVectorQuantity("Parallel Transport with one vector", pchTransport, pBasisX, pBasisY);
  psMesh->addVertexTangentVectorQuantity("Parallel Transport with sources points", pchTransportSourcesPoints, pBasisX, pBasisY);
  // psMesh->addVertexTangentVectorQuantity("Parallel Transport with mesh", vhmTransport, vBasisX, vBasisY);
  psMesh->addVertexTangentVectorQuantity("Sources Points", field, pBasisX, pBasisY);
  psMesh->addVertexTangentVectorQuantity("Parallel Transport with Random Vectors", pchTransportRandSources, pBasisX, pBasisY);
  psMesh->addVertexTangentVectorQuantity("Único Vec", initialVec, pBasisX, pBasisY); 
  psMesh->addVertexTangentVectorQuantity("Random Vectors", fieldrand, pBasisX, pBasisY);
  // psMesh->addVertexTangentVectorQuantity("X0", X0, pBasisX, pBasisY);
  // psMesh->addVertexTangentVectorQuantity("X", X, pBasisX, pBasisY);
  // psMesh->addVertexTangentVectorQuantity("Standard Smoothing", pchTransportsm, pBasisX, pBasisY);
  // psMesh->addVertexTangentVectorQuantity("0.001 Smoothing", pchTransportsmo, pBasisX, pBasisY);
  psMesh->addVertexVectorQuantity("Normal", pNormals);
  psMesh->addVertexVectorQuantity("Basis X", pBasisX);
  psMesh->addVertexVectorQuantity("Basis Y", pBasisY);
  psMesh->addVertexScalarQuantity("Error", error);
  // psMesh->addVertexScalarQuantity("Smoothing Error", smoError);

  // Give control to the polyscope gui
  polyscope::show();

  return EXIT_SUCCESS;
}
