#include <float.h>
#include <assert.h>
#include "meshEdit.h"
#include "mutablePriorityQueue.h"
#include "error_dialog.h"

namespace CMU462 {

VertexIter HalfedgeMesh::splitEdge(EdgeIter e0) {
  // TODO: (meshEdit)
  // This method should split the given edge and return an iterator to the
  // newly inserted vertex. The halfedge of this vertex should point along
  // the edge that was split, rather than the new edges.

//  showError("splitEdge() not implemented.");

	
	if (e0->isBoundary())
	{
		// TODO: BOUNDARY SPLIT!!! NOT FINISHED!
		//return e0->halfedge()->vertex();

		HalfedgeIter h0 = e0->halfedge();
		HalfedgeIter h1 = h0->next();
		HalfedgeIter h2 = h1->next();

		if (h2->next() != h0)
		{
			showError("splitEdge is not on a triangle mesh!");
			return e0->halfedge()->vertex();
		}

		HalfedgeIter h3 = newHalfedge();
		HalfedgeIter h4 = newHalfedge();
		HalfedgeIter h5 = newHalfedge();

		VertexIter v0 = h0->vertex();
		VertexIter v1 = h1->vertex();
		VertexIter v2 = h2->vertex();

		VertexIter v3 = newVertex();
		

		EdgeIter e1 = newEdge();
		EdgeIter e2 = newEdge();

		
		FaceIter f0 = h0->face();
		FaceIter f1 = newBoundary();

		h0->next() = h3;
		//h0->twin() = h1;
		//h0->vertex() = v4;
		//h0->edge() = e0;
		//h0->face() = f0;

		h1->next() = h5;
		//h1->twin() = h1;
		//h1->vertex() = v4;
		//h1->edge() = e0;
		h1->face() = f1;

		//h2->next() = h3;
		//h2->twin() = h1;
		//h2->vertex() = v4;
		//h2->edge() = e0;
		//h2->face() = f0;

		h3->next() = h2;
		h3->twin() = h5;
		h3->vertex() = v3;
		h3->edge() = e2;
		h3->face() = f0;

		h4->next() = h1;
		//h4->twin() = h5;
		h4->vertex() = v3;
		h4->edge() = e1;
		h4->face() = f1;

		h5->next() = h4;
		h5->twin() = h3;
		h5->vertex() = v2;
		h5->edge() = e2;
		h5->face() = f1;

		v0->halfedge() = h0;
		v1->halfedge() = h1;
		v2->halfedge() = h2;
		v3->halfedge() = h3;

		v3->position = v3->neighborhoodCentroid();


		e0->halfedge() = h0;
		e1->halfedge() = h4;
		e2->halfedge() = h5;

		f0->halfedge() = h0;
		f1->halfedge() = h1;

		//f1->_isBoundary = true;
		return v3;
		//
	}
		



	// HALFEDGES
	HalfedgeIter h0 = e0->halfedge();
	HalfedgeIter h1 = h0->twin();
	HalfedgeIter h2 = h0->next();
	HalfedgeIter h3 = h2->next();

	HalfedgeIter h4 = h1->next();
	HalfedgeIter h5 = h4->next();


	// check triangle mesh
	if (h5->next() != h1 || h3->next() != h0)
	{
		showError("splitEdge is not on a triangle mesh!");
		return e0->halfedge()->vertex();
	}

	HalfedgeIter h6 = newHalfedge();
	HalfedgeIter h7 = newHalfedge();
	HalfedgeIter h8 = newHalfedge();
	HalfedgeIter h9 = newHalfedge();
	HalfedgeIter h10 = newHalfedge();
	HalfedgeIter h11 = newHalfedge();
	
	


	
	// VERTICES
	VertexIter v0 = h2->vertex();
	VertexIter v1 = h3->vertex();
	VertexIter v2 = h4->vertex();
	VertexIter v3 = h5->vertex();

	VertexIter v4 = newVertex();


	// EDGES
	EdgeIter e1 = newEdge();
	EdgeIter e2 = newEdge();
	EdgeIter e3 = newEdge();

	// FACES
	FaceIter f0 = h0->face();
	FaceIter f1 = h1->face();
	FaceIter f2 = newFace();
	FaceIter f3 = newFace();

	
	// REASSIGN HERE
	// HALFEDGES
	//h0->next() = h5;
	//h0->twin() = h1;
	h0->vertex() = v4;
	//h0->edge() = e0;
	//h0->face() = f0;
	
	h1->next() = h6;
	//h1->twin() = h1;
	//h1->vertex() = v4;
	//h1->edge() = e0;
	//h1->face() = f0;
	
	h2->next() = h7;
	//h2->twin() = h1;
	//h2->vertex() = v4;
	//h2->edge() = e0;
	//h2->face() = f0;
	
	h3->next() = h8;
	//h3->twin() = h1;
	//h3->vertex() = v4;
	//h3->edge() = e0;
	h3->face() = f2;

	h4->next() = h10;
	//h4->twin() = h1;
	//h4->vertex() = v4;
	//h4->edge() = e0;
	h4->face() = f3;

	//h5->next() = h8;
	//h5->twin() = h1;
	//h5->vertex() = v4;
	//h5->edge() = e0;
	//h5->face() = f2;
	//printf("here\n");
	h6->next() = h5;
	//printf("here2\n");
	h6->twin() = h10;
	h6->vertex() = v4;
	h6->edge() = e3;
	h6->face() = f1;
	
	h7->next() = h0;
	h7->twin() = h9;
	h7->vertex() = v1;
	h7->edge() = e1;
	h7->face() = f0;

	h8->next() = h9;
	h8->twin() = h11;
	h8->vertex() = v2;
	h8->edge() = e2;
	h8->face() = f2;

	h9->next() = h3;
	h9->twin() = h7;
	h9->vertex() = v4;
	h9->edge() = e1;
	h9->face() = f2;

	h10->next() = h11;
	h10->twin() = h6;
	h10->vertex() = v3;
	h10->edge() = e3;
	h10->face() = f3;

	h11->next() = h4;
	h11->twin() = h8;
	h11->vertex() = v4;
	h11->edge() = e2;
	h11->face() = f3;


	
	// VERTICES
	v0->halfedge() = h1;
	v1->halfedge() = h3;
	v2->halfedge() = h4;
	v3->halfedge() = h5;
	v4->halfedge() = h6;

	v4->position = (v0->position + v2->position) / 2.f;

	// EDGES
	e0->halfedge() = h0; 
	e1->halfedge() = h9;
	e2->halfedge() = h11;
	e3->halfedge() = h6;


	// FACES
	f0->halfedge() = h0; 
	f1->halfedge() = h1;
	f2->halfedge() = h3;
	f3->halfedge() = h4;
	
	//checkConsistency();

	return v4;
}

void HalfedgeMesh::mergeEdges(HalfedgeIter h1, HalfedgeIter h2)
{
	HalfedgeIter h3 = h1->twin();
	HalfedgeIter h4 = h2->twin();

	VertexIter v0 = h1->vertex();
	VertexIter v1 = h2->vertex();


	HalfedgeIter h6 = h3->next();
	HalfedgeIter h5 = h6;
	for (h5 = h6; h5->twin()->vertex() != v1; h5 = h5->next()) {};

	HalfedgeIter h8 = h4->next();
	HalfedgeIter h7 = h8;
	for (h7 = h8; h7->twin()->vertex() != v0; h7 = h7->next()) {};



	EdgeIter e0 = h2->edge();	// remain
	EdgeIter e1 = h1->edge();

	FaceIter f0 = h3->face();
	FaceIter f1 = h4->face();
	FaceIter f3 = h1->face();

	h5->next() = h2;
	
	h2->next() = h6;
	h2->twin() = h1;
	h2->face() = f0;

	h7->next() = h1;

	h1->next() = h8;
	h1->twin() = h2;
	h1->edge() = e0;
	h1->face() = f1;


	e0->halfedge() = h2;

	v0->halfedge() = h1;
	v1->halfedge() = h2;


	f0->halfedge() = h2;
	f1->halfedge() = h1;

	deleteEdge(e1);
	deleteHalfedge(h3);
	deleteHalfedge(h4);
	deleteFace(f3);


}

VertexIter HalfedgeMesh::collapseEdge(EdgeIter e) {
  // TODO: (meshEdit)
  // This method should collapse the given edge and return an iterator to
  // the new vertex created by the collapse.
	if (e->isBoundary())
	{
		return e->halfedge()->vertex();
	}
		

	HalfedgeIter h0 = e->halfedge();
	HalfedgeIter h1 = h0->next();
	HalfedgeIter h2;
	for (h2 = h1; h2->next() != h0; h2 = h2->next()) {};
	

	VertexIter v0 = h0->vertex();

	FaceIter f0 = h0->face();




	HalfedgeIter h3 = h0->twin();
	HalfedgeIter h4 = h3->next();
	HalfedgeIter h5;
	VertexIter v1 = h3->vertex();
	for (h5 = h4; h5->twin()->vertex() != v1; h5 = h5->next()) { };

	
	
	FaceIter f1 = h3->face();
	

	// Change all the origin from v1
	HalfedgeIter h = h3;
	do
	{
		h->vertex() = v0;
		h = h->twin()->next();
	} while (h != h3);



	
	h2->next() = h1;
	h5->next() = h4;

	v0->halfedge() = h1;

	f0->halfedge() = h2;
	f1->halfedge() = h4;

	/* Delete*/
	deleteEdge(e);
	deleteHalfedge(h0);
	deleteHalfedge(h3);
	deleteVertex(v1);

	//checkConsistency();
	/* If triangle, merge edge */
	if (h1->next() == h2)
		mergeEdges(h1, h2);
	//
	if (h4->next() == h5)
		mergeEdges(h4, h5);

	v0->position = v0->neighborhoodCentroid();
	checkConsistency();

//  showError("collapseEdge() not implemented.");
  return v0;
}

VertexIter HalfedgeMesh::collapseFace(FaceIter f) {
  // TODO: (meshEdit)
  // This method should collapse the given face and return an iterator to
  // the new vertex created by the collapse.
  showError("collapseFace() not implemented.");
  return f->halfedge()->vertex();
}

FaceIter HalfedgeMesh::eraseVertex(VertexIter v) {
  // TODO: (meshEdit)
  // This method should replace the given vertex and all its neighboring
  // edges and faces with a single face, returning the new face.

	if (v->isBoundary())
	{
		showError("Can not erase a boundary vertex!");
		return v->halfedge()->face();
	}
	HalfedgeIter H0 = v->halfedge();
	HalfedgeIter h0 = H0->twin()->next();
	HalfedgeIter next;
	HalfedgeIter H3;
	for (H3 = h0; H3->next()->twin() != H0; H3 = H3->next());
	VertexIter v0 = v;

	FaceIter F0 = H0->twin()->face();

	while (h0 != H0)
	{
		
		next = h0->twin()->next();
		HalfedgeIter last = h0->next();
		HalfedgeIter h1 = h0->twin();
		HalfedgeIter h2 = h1->next();
		HalfedgeIter h3;

		for (h3 = h2; h3->next() != h1; h3 = h3->next())
		{
			h3->face() = F0;
		}
		h3->next() = last;
		h3->face() = F0;


		VertexIter v1 = h1->vertex();

		EdgeIter e0 = h0->edge();

		FaceIter f0 = h1->face();



		v1->halfedge() = last;

		deleteHalfedge(h0);
		deleteHalfedge(h1);

		deleteEdge(e0);
		deleteFace(f0);

		h0 = next;
	}

	// last edge
	HalfedgeIter last = h0->next();
	HalfedgeIter h1 = h0->twin();

	H3->next() = last;
	H3->face() = F0;

	VertexIter v1 = h1->vertex();

	EdgeIter e0 = h0->edge();

	v1->halfedge() = last;

	deleteHalfedge(h0);
	deleteHalfedge(h1);

	deleteEdge(e0);
	deleteVertex(v0);

	F0->halfedge() = H3;

	checkConsistency();
	return F0;
  //return FaceIter();
}

FaceIter HalfedgeMesh::eraseEdge(EdgeIter e) {
  // TODO: (meshEdit)
  // This method should erase the given edge and return an iterator to the
  // merged face.

  showError("eraseVertex() not implemented.");
  return FaceIter();
}

EdgeIter HalfedgeMesh::flipEdge(EdgeIter e0) {
  // TODO: (meshEdit)
  // This method should flip the given edge and return an iterator to the
  // flipped edge.
	
	// corner case
	if (e0->isBoundary())
	{
		showError("flipEdge() on a boundary edge.");
		return e0;
	}
		
		
	
	
	// HALFEDGES
	HalfedgeIter h0 = e0->halfedge();
	HalfedgeIter h1 = h0->twin();
	HalfedgeIter h2 = h1->next();
	
	HalfedgeIter h4 = h0->next();
	HalfedgeIter h5 = h4->next();
	HalfedgeIter h6 = h2->next();

	// VERTICES
	VertexIter v0 = h0->vertex();
	VertexIter v1 = h2->twin()->vertex();
	VertexIter v2 = h1->vertex();
	VertexIter v3 = h4->twin()->vertex();
	

	HalfedgeIter h3;
	for (h3 = h2; h3->twin()->vertex() != v2; h3 = h3->next()) {};
	HalfedgeIter h7;
	for (h7 = h4; h7->twin()->vertex() != v0; h7 = h7->next()) {};

	//// EDGES
	//EdgeIter e1 = h5->edge();
	//EdgeIter e2 = h4->edge();
	//EdgeIter e3 = h2->edge(); 
	//EdgeIter e4 = h1->edge(); 


	// FACES
	FaceIter f0 = h0->face();
	FaceIter f1 = h1->face();



	// REASSIGN HERE
	// HALFEDGES
	h0->next() = h5;
	h0->twin() = h1;
	h0->vertex() = v1;
//	h0->edge() = e0;
//	h0->face() = f0;

	h1->next() = h6;
	h1->twin() = h0;
	h1->vertex() = v3;
//	h1->edge() = e3;
//	h1->face() = f0;

	h2->next() = h0;
//	h2->twin() = h8;
//	h2->vertex() = v0;
//	h2->edge() = e2;
	h2->face() = f0;
	
	h3->next() = h4;
//	h3->twin() = h0;
//	h3->vertex() = v3;
//	h3->edge() = e0;
//	h3->face() = f1;

	h4->next() = h1;
//	h4->twin() = h9;
//	h4->vertex() = v2;
//	h4->edge() = e1;
	h4->face() = f1;

	h7->next() = h2;


	// VERTICES
	v0->halfedge() = h2;
	v1->halfedge() = h0;
	v2->halfedge() = h4;
	v3->halfedge() = h5;	

	//// EDGES
	e0->halfedge() = h0; //...you fill in the rest!...
	//e1->halfedge() = h4;
	//e2->halfedge() = h2;
	//e3->halfedge() = h1;
	//e4->halfedge() = h5;

	// FACES
	f0->halfedge() = h0; //...you fill in the rest!...
	f1->halfedge() = h1;

//	checkConsistency();
	
  return e0;
}

void HalfedgeMesh::subdivideQuad(bool useCatmullClark) {
  // Unlike the local mesh operations (like bevel or edge flip), we will perform
  // subdivision by splitting *all* faces into quads "simultaneously."  Rather
  // than operating directly on the halfedge data structure (which as you've
  // seen
  // is quite difficult to maintain!) we are going to do something a bit nicer:
  //
  //    1. Create a raw list of vertex positions and faces (rather than a full-
  //       blown halfedge mesh).
  //
  //    2. Build a new halfedge mesh from these lists, replacing the old one.
  //
  // Sometimes rebuilding a data structure from scratch is simpler (and even
  // more
  // efficient) than incrementally modifying the existing one.  These steps are
  // detailed below.

  // TODO Step I: Compute the vertex positions for the subdivided mesh.  Here
  // we're
  // going to do something a little bit strange: since we will have one vertex
  // in
  // the subdivided mesh for each vertex, edge, and face in the original mesh,
  // we
  // can nicely store the new vertex *positions* as attributes on vertices,
  // edges,
  // and faces of the original mesh.  These positions can then be conveniently
  // copied into the new, subdivided mesh.
  // [See subroutines for actual "TODO"s]
  if (useCatmullClark) {
    computeCatmullClarkPositions();
  } else {
    computeLinearSubdivisionPositions();
  }

  // TODO Step II: Assign a unique index (starting at 0) to each vertex, edge,
  // and
  // face in the original mesh.  These indices will be the indices of the
  // vertices
  // in the new (subdivided mesh).  They do not have to be assigned in any
  // particular
  // order, so long as no index is shared by more than one mesh element, and the
  // total number of indices is equal to V+E+F, i.e., the total number of
  // vertices
  // plus edges plus faces in the original mesh.  Basically we just need a
  // one-to-one
  // mapping between original mesh elements and subdivided mesh vertices.
  // [See subroutine for actual "TODO"s]
  assignSubdivisionIndices();

  // TODO Step III: Build a list of quads in the new (subdivided) mesh, as
  // tuples of
  // the element indices defined above.  In other words, each new quad should be
  // of
  // the form (i,j,k,l), where i,j,k and l are four of the indices stored on our
  // original mesh elements.  Note that it is essential to get the orientation
  // right
  // here: (i,j,k,l) is not the same as (l,k,j,i).  Indices of new faces should
  // circulate in the same direction as old faces (think about the right-hand
  // rule).
  // [See subroutines for actual "TODO"s]
  vector<vector<Index> > subDFaces;
  vector<Vector3D> subDVertices;
  buildSubdivisionFaceList(subDFaces);
  buildSubdivisionVertexList(subDVertices);

  // TODO Step IV: Pass the list of vertices and quads to a routine that clears
  // the
  // internal data for this halfedge mesh, and builds new halfedge data from
  // scratch,
  // using the two lists.
  rebuild(subDFaces, subDVertices);
}

/**
 * Compute new vertex positions for a mesh that splits each polygon
 * into quads (by inserting a vertex at the face midpoint and each
 * of the edge midpoints).  The new vertex positions will be stored
 * in the members Vertex::newPosition, Edge::newPosition, and
 * Face::newPosition.  The values of the positions are based on
 * simple linear interpolation, e.g., the edge midpoints and face
 * centroids.
 */
void HalfedgeMesh::computeLinearSubdivisionPositions() {
  // TODO For each vertex, assign Vertex::newPosition to
  // its original position, Vertex::position.
	for (VertexIter v = verticesBegin(); v != verticesEnd(); v++)
	{
		v->newPosition = v->position;
	}


  // TODO For each edge, assign the midpoint of the two original
  // positions to Edge::newPosition.
	for (EdgeIter e = edgesBegin(); e != edgesEnd(); e++)
	{
		HalfedgeIter h = e->halfedge();
		e->newPosition = (h->vertex()->position + h->twin()->vertex()->position) / 2.f;
	}


  // TODO For each face, assign the centroid (i.e., arithmetic mean)
  // of the original vertex positions to Face::newPosition.  Note
  // that in general, NOT all faces will be triangles!
	for (FaceIter f = facesBegin(); f != facesEnd(); f++)
	{
		Vector3D pos = Vector3D();
		HalfedgeIter h = f->halfedge();
		double count = 0;
		do
		{
			pos += h->vertex()->position;
			count += 1.f;
			h = h->next();
		} while (h != f->halfedge());

		pos /= count;
		f->newPosition = pos;

	}


  //showError("computeLinearSubdivisionPositions() not implemented.");
}

/**
 * Compute new vertex positions for a mesh that splits each polygon
 * into quads (by inserting a vertex at the face midpoint and each
 * of the edge midpoints).  The new vertex positions will be stored
 * in the members Vertex::newPosition, Edge::newPosition, and
 * Face::newPosition.  The values of the positions are based on
 * the Catmull-Clark rules for subdivision.
 */
void HalfedgeMesh::computeCatmullClarkPositions() {
  // TODO The implementation for this routine should be
  // a lot like HalfedgeMesh::computeLinearSubdivisionPositions(),
  // except that the calculation of the positions themsevles is
  // slightly more involved, using the Catmull-Clark subdivision
  // rules. (These rules are outlined in the Developer Manual.)

  // TODO face
	for (FaceIter f = facesBegin(); f != facesEnd(); f++)
	{
		Vector3D pos = Vector3D();
		HalfedgeIter h = f->halfedge();
		double count = 0;
		do
		{
			pos += h->vertex()->position;
			count += 1.f;
			h = h->next();
		} while (h != f->halfedge());

		pos /= count;
		f->newPosition = pos;

	}
  // TODO edges
	for (EdgeIter e = edgesBegin(); e != edgesEnd(); e++)
	{
		HalfedgeIter h = e->halfedge();
		e->newPosition = (h->face()->newPosition + h->twin()->face()->newPosition) / 2.f;
	}

  // TODO vertices
	for (VertexIter v = verticesBegin(); v != verticesEnd(); v++)
	{
		// Compute face average
		Vector3D Q = Vector3D();
		Vector3D R = Vector3D();
		double count = 0;
		HalfedgeIter h = v->halfedge();
		do
		{
			Q += h->face()->newPosition;

			R = R + (h->vertex()->position + h->twin()->vertex()->position) / 2.f;

			count += 1.f;
			h = h->twin()->next();
		} while (h != v->halfedge());
		
		Q /= count;
		R /= count;

		Vector3D S = v->position;

		v->newPosition = (Q + 2 * R + (count - 3) * S) / count;
		//v->newPosition = v->position;
	}
  
  //  showError("computeCatmullClarkPositions() not implemented.");
}

/**
 * Assign a unique integer index to each vertex, edge, and face in
 * the mesh, starting at 0 and incrementing by 1 for each element.
 * These indices will be used as the vertex indices for a mesh
 * subdivided using Catmull-Clark (or linear) subdivision.
 */
void HalfedgeMesh::assignSubdivisionIndices() {
  // TODO Start a counter at zero; if you like, you can use the
  // "Index" type (defined in halfedgeMesh.h)
	int count = 0;
  // TODO Iterate over vertices, assigning values to Vertex::index
	for (VertexIter v = verticesBegin(); v != verticesEnd(); v++)
	{
		v->index = count++;
	}
	// TODO Iterate over edges, assigning values to Edge::index
	for (EdgeIter e = edgesBegin(); e != edgesEnd(); e++)
	{
		e->index = count++;
	}


	// TODO Iterate over faces, assigning values to Face::index
	for (FaceIter f = facesBegin(); f != facesEnd(); f++)
	{
		f->index = count++;
	}
  

 
 // showError("assignSubdivisionIndices() not implemented.");
}

/**
 * Build a flat list containing all the vertex positions for a
 * Catmull-Clark (or linear) subdivison of this mesh.  The order of
 * vertex positions in this list must be identical to the order
 * of indices assigned to Vertex::newPosition, Edge::newPosition,
 * and Face::newPosition.
 */
void HalfedgeMesh::buildSubdivisionVertexList(vector<Vector3D>& subDVertices) {
  // TODO Resize the vertex list so that it can hold all the vertices.
	//?
	subDVertices.clear();
  // TODO Iterate over vertices, assigning Vertex::newPosition to the
  // appropriate location in the new vertex list.

	for (VertexIter v = verticesBegin(); v != verticesEnd(); v++)
	{
		subDVertices.push_back(v->newPosition);
	}
  // TODO Iterate over edges, assigning Edge::newPosition to the appropriate
  // location in the new vertex list.
	for (EdgeIter e = edgesBegin(); e != edgesEnd(); e++)
	{
		subDVertices.push_back(e->newPosition);
	}


	// TODO For each face, assign the centroid (i.e., arithmetic mean)
	// of the original vertex positions to Face::newPosition.  Note
	// that in general, NOT all faces will be triangles!
	for (FaceIter f = facesBegin(); f != facesEnd(); f++)
	{
		subDVertices.push_back(f->newPosition);
	}

  // TODO Iterate over faces, assigning Face::newPosition to the appropriate
  // location in the new vertex list.
//  showError("buildSubdivisionVertexList() not implemented.");
}

/**
 * Build a flat list containing all the quads in a Catmull-Clark
 * (or linear) subdivision of this mesh.  Each quad is specified
 * by a vector of four indices (i,j,k,l), which come from the
 * members Vertex::index, Edge::index, and Face::index.  Note that
 * the ordering of these indices is important because it determines
 * the orientation of the new quads; it is also important to avoid
 * "bowties."  For instance, (l,k,j,i) has the opposite orientation
 * of (i,j,k,l), and if (i,j,k,l) is a proper quad, then (i,k,j,l)
 * will look like a bowtie.
 */
void HalfedgeMesh::buildSubdivisionFaceList(vector<vector<Index> >& subDFaces) {
  // TODO This routine is perhaps the most tricky step in the construction of
  // a subdivision mesh (second, perhaps, to computing the actual Catmull-Clark
  // vertex positions).  Basically what you want to do is iterate over faces,
  // then for each for each face, append N quads to the list (where N is the
  // degree of the face).  For this routine, it may be more convenient to simply
  // append quads to the end of the list (rather than allocating it ahead of
  // time), though YMMV.  You can of course iterate around a face by starting
  // with its first halfedge and following the "next" pointer until you get
  // back to the beginning.  The tricky part is making sure you grab the right
  // indices in the right order---remember that there are indices on vertices,
  // edges, AND faces of the original mesh.  All of these should get used.  Also
  // remember that you must have FOUR indices per face, since you are making a
  // QUAD mesh!

	int edgeStart = nVertices();
	int faceStart = edgeStart + nEdges();
	int fcount = 0;
  // TODO iterate over faces
	for (FaceIter f = facesBegin(); f != facesEnd(); f++)
	{
		Index center = faceStart + fcount;

		
		
		// TODO loop around face
		HalfedgeIter H0 = f->halfedge();
		HalfedgeIter h0 = H0;

		do
		{
			vector<Index> quad(4);
			// TODO build lists of four indices for each sub-quad
			// push center
			quad[0] = center;

			HalfedgeIter h1 = h0->next();
			// find mid of h0
			quad[1] = h0->edge()->index;

			// find v
			VertexIter v = h1->vertex();
			quad[2] = v->index;

			// find mid of h1
			quad[3] = h1->edge()->index;

			// TODO append each list of four indices to face list
			subDFaces.push_back(quad);
			h0 = h0->next();
		} while (h0 != H0);
		
		
		
		fcount++;
	}
//	showError("buildSubdivisionFaceList() not implemented.");
}

FaceIter HalfedgeMesh::bevelVertex(VertexIter v) {
  // TODO This method should replace the vertex v with a face, corresponding to
  // a bevel operation. It should return the new face.  NOTE: This method is
  // responsible for updating the *connectivity* of the mesh only---it does not
  // need to update the vertex positions.  These positions will be updated in
  // HalfedgeMesh::bevelVertexComputeNewPositions (which you also have to
  // implement!)

  showError("bevelVertex() not implemented.");
  return facesBegin();
}

FaceIter HalfedgeMesh::bevelEdge(EdgeIter e) {
  // TODO This method should replace the edge e with a face, corresponding to a
  // bevel operation. It should return the new face.  NOTE: This method is
  // responsible for updating the *connectivity* of the mesh only---it does not
  // need to update the vertex positions.  These positions will be updated in
  // HalfedgeMesh::bevelEdgeComputeNewPositions (which you also have to
  // implement!)

  showError("bevelEdge() not implemented.");
  return facesBegin();
}

FaceIter HalfedgeMesh::bevelFace(FaceIter f) {
  // TODO This method should replace the face f with an additional, inset face
  // (and ring of faces around it), corresponding to a bevel operation. It
  // should return the new face.  NOTE: This method is responsible for updating
  // the *connectivity* of the mesh only---it does not need to update the vertex
  // positions.  These positions will be updated in
  // HalfedgeMesh::bevelFaceComputeNewPositions (which you also have to
  // implement!)

  //showError("bevelFace() not implemented.");
	HalfedgeIter H0 = f->halfedge();
	HalfedgeIter H3 = newHalfedge();
	HalfedgeIter H6 = newHalfedge();

	VertexIter V0 = H0->vertex();
	VertexIter V3 = newVertex();

	EdgeIter E3 = newEdge();

	H3->next() = H0;
	H3->twin() = H6;
	H3->vertex() = V3;
	H3->edge() = E3;
	// TODO FACE LOOP

	// TODO H6 NEXT!
	H6->twin() = H3;
	H6->vertex() = V0;
	H6->edge() = E3;
	// TODO H6 FACE

	V3->halfedge() = H3;
	E3->halfedge() = H3;

	HalfedgeIter h0 = H0;
	HalfedgeIter next = h0->next();
	HalfedgeIter prev;

	FaceIter F0 = H0->face();

	VertexIter v3 = V3;
	while (h0->next() != H0)
	{
		next = h0->next();


		HalfedgeIter h1 = newHalfedge();
		HalfedgeIter h2 = newHalfedge();
		HalfedgeIter h3 = v3->halfedge();
		HalfedgeIter h4 = newHalfedge();
		HalfedgeIter h5 = newHalfedge();
		
		VertexIter v0 = h0->vertex();
		VertexIter v1 = h0->twin()->vertex();
		VertexIter v2 = newVertex();

		EdgeIter e1 = newEdge();
		EdgeIter e2 = newEdge();

		FaceIter f0 = newFace();

		h0->next() = h1;
		h0->face() = f0;

		h1->next() = h2;
		h1->twin() = h4;
		h1->vertex() = v1;
		h1->edge() = e1;
		h1->face() = f0;

		h2->next() = h3;
		h2->twin() = h5;
		h2->vertex() = v2;
		h2->edge() = e2;
		h2->face() = f0;

		h3->vertex() = v3;
		h3->face() = f0;

		h4->next() = next;
		h4->twin() = h1;
		h4->vertex() = v2;
		h4->edge() = e1;
		//h4->face() = f0;

		if (h0 != H0)
			prev->next() = h5;
		//h5->next() = h3;
		h5->twin() = h2;
		h5->vertex() = v3;
		h5->edge() = e2;
		h5->face() = F0;

		v2->halfedge() = h4;

		e1->halfedge() = h1;
		e2->halfedge() = h2;

		f0->halfedge() = h0;

		

		v3 = v2;
		h0 = next;
		prev = h5;
	}
	// last remain

	VertexIter v0 = h0->vertex();
	VertexIter v1 = h0->twin()->vertex();
	

	HalfedgeIter h1 = H6;
	HalfedgeIter h2 = newHalfedge();
	HalfedgeIter h3 = v3->halfedge();
	HalfedgeIter h4 = h1->twin();
	HalfedgeIter h5 = newHalfedge();

	VertexIter v2 = h4->vertex();

	EdgeIter e1 = h1->edge();
	EdgeIter e2 = newEdge();

	FaceIter f0 = newFace();

	h0->next() = h1;
	h0->face() = f0;

	h1->next() = h2;
	h1->face() = f0;

	h2->next() = h3;
	h2->twin() = h5;
	h2->vertex() = v2;
	h2->edge() = e2;
	h2->face() = f0;

	h3->face() = f0;

	prev->next() = h5;
	h5->next() = h4->next()->next()->next()->twin();
	h5->twin() = h2;
	h5->vertex() = v3;
	h5->edge() = e2;
	h5->face() = F0;

	e2->halfedge() = h2;

	f0->halfedge() = h0;

	// inner face
	F0->halfedge() = h5;

	HalfedgeIter b;
	HalfedgeIter c;
	HalfedgeIter h = h5;

	Vector3D sum = Vector3D();
	double count = 0;
	do
	{
		sum = sum + h->twin()->next()->next()->vertex()->position;
		count += 1.f;

		h = h->next();
	} while (h != h5);
	
	sum /= count;
	
	h = h5;
	do
	{
		Vector3D pb = h->twin()->next()->twin()->vertex()->position;

		//newHalfedges[b]->vertex()->position = pb/2;
		Vector3D pos = (pb + sum)/2.f;
		h->vertex()->position = pos;
		//printf("%f %f %f	%f %f %f	%f %f %f\n", sum.x, sum.y, sum.z, pb.x, pb.y, pb.z, pos.x, pos.y, pos.z);
		h = h->next();
	} while (h != h5);


	checkConsistency();

  return F0;
}


void HalfedgeMesh::bevelFaceComputeNewPositions(
    vector<Vector3D>& originalVertexPositions,
    vector<HalfedgeIter>& newHalfedges, double normalShift,
    double tangentialInset) {
  // TODO Compute new vertex positions for the vertices of the beveled face.
  //
  // These vertices can be accessed via newHalfedges[i]->vertex()->position for
  // i = 1, ..., newHalfedges.size()-1.
  //
  // The basic strategy here is to loop over the list of outgoing halfedges,
  // and use the preceding and next vertex position from the original mesh
  // (in the originalVertexPositions array) to compute an offset vertex
  // position.
  //
  // Note that there is a 1-to-1 correspondence between halfedges in
  // newHalfedges and vertex positions
  // in orig.  So, you can write loops of the form
  //
  // for( int i = 0; i < newHalfedges.size(); hs++ )
  // {
  //    Vector3D pi = originalVertexPositions[i]; // get the original vertex
  //    position correponding to vertex i
  // }
  //


	int N = newHalfedges.size();
	//printf("%d\n",N);
	FaceIter f0 = newHalfedges[0]->twin()->next()->twin()->face();
	const Vector3D faceNorm = f0->normal();
	printf("%f %f %f : %f %f\n", faceNorm.x, faceNorm.y, faceNorm.z, normalShift, tangentialInset);

	// find center
	Vector3D center = Vector3D();
	for(int i = 0; i < N; ++i)
	{
		center = center + newHalfedges[i]->vertex()->position;
	}

	double count = N;
	center /= count;


	for (int i = 0; i < newHalfedges.size(); ++i)
	{
		int a = (i + N - 1) % N;
		int b = i;
		int c = (i + 1) % N;

		// Get the actual 3D vertex coordinates at these vertices
		//Vector3D pa = originalVertexPositions[a];
		//Vector3D pb = originalVertexPositions[b];
		//Vector3D pc = originalVertexPositions[c];
		Vector3D pa = newHalfedges[a]->twin()->vertex()->position;
		Vector3D pb = newHalfedges[b]->twin()->vertex()->position;
		Vector3D pc = newHalfedges[c]->twin()->vertex()->position;

		
		Vector3D pos = newHalfedges[b]->vertex()->position + center * tangentialInset - normalShift * faceNorm ;
		//Vector3D pos = (pb+center)/2.f +normalShift * faceNorm;
		newHalfedges[b]->vertex()->position = pos;
	
	}


}

void HalfedgeMesh::bevelVertexComputeNewPositions(
    Vector3D originalVertexPosition, vector<HalfedgeIter>& newHalfedges,
    double tangentialInset) {
  // TODO Compute new vertex positions for the vertices of the beveled vertex.
  //
  // These vertices can be accessed via newHalfedges[i]->vertex()->position for
  // i = 1, ..., hs.size()-1.
  //
  // The basic strategy here is to loop over the list of outgoing halfedges,
  // and use the preceding and next vertex position from the original mesh
  // (in the orig array) to compute an offset vertex position.

}

void HalfedgeMesh::bevelEdgeComputeNewPositions(
    vector<Vector3D>& originalVertexPositions,
    vector<HalfedgeIter>& newHalfedges, double tangentialInset) {
  // TODO Compute new vertex positions for the vertices of the beveled edge.
  //
  // These vertices can be accessed via newHalfedges[i]->vertex()->position for
  // i = 1, ..., newHalfedges.size()-1.
  //
  // The basic strategy here is to loop over the list of outgoing halfedges,
  // and use the preceding and next vertex position from the original mesh
  // (in the orig array) to compute an offset vertex position.
  //
  // Note that there is a 1-to-1 correspondence between halfedges in
  // newHalfedges and vertex positions
  // in orig.  So, you can write loops of the form
  //
  // for( int i = 0; i < newHalfedges.size(); i++ )
  // {
  //    Vector3D pi = originalVertexPositions[i]; // get the original vertex
  //    position correponding to vertex i
  // }
  //

}

void HalfedgeMesh::splitPolygons(vector<FaceIter>& fcs) {
  for (auto f : fcs) splitPolygon(f);
}

void HalfedgeMesh::splitPolygon(FaceIter f) {
  // TODO: (meshedit) 
  // Triangulate a polygonal face
  //showError("splitPolygon() not implemented.");
	// for each face

	HalfedgeIter H0 = f->halfedge();

	// Return if its a triangle
	if (H0->next()->next()->next() == H0)
		return;

	VertexIter v0 = H0->vertex();

	HalfedgeIter h2 = H0;
	HalfedgeIter h0 = H0->next();
	HalfedgeIter next;
	while (h0->next()->twin()->vertex() != v0)
	{
		next = h0->next();
		HalfedgeIter h1 = newHalfedge();
		HalfedgeIter h3 = newHalfedge();
		HalfedgeIter h4 = h2->twin();

		VertexIter v1 = h4->vertex();
		VertexIter v2 = h0->twin()->vertex();

		EdgeIter e0 = newEdge();
		
		FaceIter f0;
		
		if (h0 == H0->next())
			f0 = h0->face();
		else
			f0 = newFace();

		h0->next() = h1;
		h0->face() = f0;

		h1->next() = h2;
		h1->twin() = h3;
		h1->vertex() = v2;
		h1->edge() = e0;
		h1->face() = f0;

		h2->next() = h0;
		h2->face() = f0;

		h3->next() = next;	// dont need
		h3->twin() = h1;
		h3->vertex() = v0;
		h3->edge() = e0;
		h3->face() = f0;

		v2->halfedge() = h1;

		e0->halfedge() = h1;

		f0->halfedge() = h0;

		h2 = h3;
		h0 = next;
	}

	// remain and not a triangle

		
		HalfedgeIter h1 = h0->next();
		HalfedgeIter h4 = h2->twin();

		VertexIter v1 = h4->vertex();
		VertexIter v2 = h1->vertex();

		FaceIter f0 = newFace();

		h0->face() = f0;

		h1->next() = h2;
		h1->face() = f0;

		h2->next() = h0;
		h2->face() = f0;

		f0->halfedge() = h0;
 	

	checkConsistency();
}

EdgeRecord::EdgeRecord(EdgeIter& _edge) : edge(_edge) {
  // TODO: (meshEdit)
  // Compute the combined quadric from the edge endpoints.
  // -> Build the 3x3 linear system whose solution minimizes the quadric error
  //    associated with these two endpoints.
  // -> Use this system to solve for the optimal position, and store it in
  //    EdgeRecord::optimalPoint.
  // -> Also store the cost associated with collapsing this edg in
  //    EdgeRecord::Cost.
}

void MeshResampler::upsample(HalfedgeMesh& mesh)
// This routine should increase the number of triangles in the mesh using Loop
// subdivision.
{
  // TODO: (meshEdit)
  // Compute new positions for all the vertices in the input mesh, using
  // the Loop subdivision rule, and store them in Vertex::newPosition.
  // -> At this point, we also want to mark each vertex as being a vertex of the
  //    original mesh.
  // -> Next, compute the updated vertex positions associated with edges, and
  //    store it in Edge::newPosition.
  // -> Next, we're going to split every edge in the mesh, in any order.  For
  //    future reference, we're also going to store some information about which
  //    subdivided edges come from splitting an edge in the original mesh, and
  //    which edges are new, by setting the flat Edge::isNew. Note that in this
  //    loop, we only want to iterate over edges of the original mesh.
  //    Otherwise, we'll end up splitting edges that we just split (and the
  //    loop will never end!)
  // -> Now flip any new edge that connects an old and new vertex.
  // -> Finally, copy the new vertex positions into final Vertex::position.

  // Each vertex and edge of the original surface can be associated with a
  // vertex in the new (subdivided) surface.
  // Therefore, our strategy for computing the subdivided vertex locations is to
  // *first* compute the new positions
  // using the connectity of the original (coarse) mesh; navigating this mesh
  // will be much easier than navigating
  // the new subdivided (fine) mesh, which has more elements to traverse.  We
  // will then assign vertex positions in
  // the new mesh based on the values we computed for the original mesh.
  // Compute updated positions for all the vertices in the original mesh, using
  // the Loop subdivision rule.

	VertexIter v;
	for (v = mesh.verticesBegin(); v != mesh.verticesEnd(); ++v)
	{
		v->isNew = false;
		int n = 0;
		HalfedgeIter h = v->halfedge();
		Vector3D sum = Vector3D();

		do
		{
			sum += h->twin()->vertex()->position;
			n++;
			h = h->twin()->next();
		} while (h != v->halfedge());

		double u = (n == 3) ? 3.0 / 16 : 3.0 / (8 * n);
		v->newPosition = sum * u + v->position * (1 - n*u);
	}


  // Next, compute the updated vertex positions associated with edges.
	EdgeIter e;
	for (e = mesh.edgesBegin(); e != mesh.edgesEnd(); ++e)
	{
		e->isNew = false;
		HalfedgeIter h = e->halfedge();
		e->newPosition = 3.0 / 8 * (h->vertex()->position + h->twin()->vertex()->position)
			+ 1.0 / 8 * (h->next()->next()->vertex()->position + h->twin()->next()->next()->vertex()->position);
	}



  // Next, we're going to split every edge in the mesh, in any order.  For
  // future
  // reference, we're also going to store some information about which
  // subdivided
  // edges come from splitting an edge in the original mesh, and which edges are
  // new.
  // In this loop, we only want to iterate over edges of the original
  // mesh---otherwise,
  // we'll end up splitting edges that we just split (and the loop will never
  // end!)
	int n = mesh.nEdges();	// origin edges
	e = mesh.edgesBegin();
	EdgeIter next;
	for (int i = 0; i < n; ++i)
	{
		next = e;
		next++;

		e->isNew = false;

		VertexIter v = mesh.splitEdge(e);
		v->isNew = true;
	
		v->position = e->newPosition;


		//set edge new, cautious: v's halfedge is on new
		v->halfedge()->edge()->isNew = true;
		v->halfedge()->twin()->next()->twin()->next()->edge()->isNew = true;

		e = next;
	}

	// Finally, flip any new edge that connects an old and new vertex.

	for (e = mesh.edgesBegin(); e != mesh.edgesEnd(); e++)
	{
		if (e->isNew)
		{
			VertexIter v0 = e->halfedge()->vertex();
			VertexIter v1 = e->halfedge()->twin()->vertex();
			if ((v0->isNew && !v1->isNew) || (!v0->isNew && v1->isNew))
			{
				mesh.flipEdge(e);
			}
		}
	}


  // Copy the updated vertex positions to the subdivided mesh.
	for (v = mesh.verticesBegin(); v != mesh.verticesEnd(); ++v)
	{
		if(!v->isNew)
			v->position = v->newPosition;
	}


//  showError("upsample() not implemented.");
}

void MeshResampler::downsample(HalfedgeMesh& mesh) {
  // TODO: (meshEdit)
  // Compute initial quadrics for each face by simply writing the plane equation
  // for the face in homogeneous coordinates. These quadrics should be stored
  // in Face::quadric
  // -> Compute an initial quadric for each vertex as the sum of the quadrics
  //    associated with the incident faces, storing it in Vertex::quadric
  // -> Build a priority queue of edges according to their quadric error cost,
  //    i.e., by building an EdgeRecord for each edge and sticking it in the
  //    queue.
  // -> Until we reach the target edge budget, collapse the best edge. Remember
  //    to remove from the queue any edge that touches the collapsing edge
  //    BEFORE it gets collapsed, and add back into the queue any edge touching
  //    the collapsed vertex AFTER it's been collapsed. Also remember to assign
  //    a quadric to the collapsed vertex, and to pop the collapsed edge off the
  //    top of the queue.
  showError("downsample() not implemented.");
}

void MeshResampler::resample(HalfedgeMesh& mesh) {
  // TODO: (meshEdit)
  // Compute the mean edge length.
  // Repeat the four main steps for 5 or 6 iterations
  // -> Split edges much longer than the target length (being careful about
  //    how the loop is written!)
  // -> Collapse edges much shorter than the target length.  Here we need to
  //    be EXTRA careful about advancing the loop, because many edges may have
  //    been destroyed by a collapse (which ones?)
  // -> Now flip each edge if it improves vertex degree
  // -> Finally, apply some tangential smoothing to the vertex positions
  showError("resample() not implemented.");
}

}  // namespace CMU462
