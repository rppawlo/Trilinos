// @HEADER
// @HEADER

#ifndef LOCA_TPETRA_LOW_RANK_UPDATE_ROW_MATRIX_DEF_HPP
#define LOCA_TPETRA_LOW_RANK_UPDATE_ROW_MATRIX_DEF_HPP

#include "LOCA_Tpetra_LowRankUpdateRowMatrix.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Vector.hpp"
#include "Tpetra_RowGraph.hpp"
#include "Tpetra_RowMatrix.hpp"
#include "Tpetra_CrsGraph.hpp"
#include "Tpetra_CrsMatrix.hpp"

namespace LOCA {
  namespace Tpetra {

    LowRankUpdateRowMatrix::
    LowRankUpdateRowMatrix(const Teuchos::RCP<LOCA::GlobalData>& global_data,
                           const Teuchos::RCP<NOX::TRowMatrix>& jacRowMatrix,
                           const Teuchos::RCP<NOX::TMultiVector>& U_multiVec,
                           const Teuchos::RCP<NOX::TMultiVector>& V_multiVec,
                           bool setup_for_solve,
                           bool include_UV_terms) :
      J_rowMatrix(jacRowMatrix),
      nonconst_U(U_multiVec),
      nonconst_V(V_multiVec),
      includeUV(include_UV_terms),
      m(U_multiVec->getNumVectors()),
      U_map(*U_multiVec->getMap()),
      V_map(*V_multiVec->getMap()),
      row_map(*jacRowMatrix->getRowMap())
    {}

    Teuchos::RCP<const Teuchos::Comm<int> >
    LowRankUpdateRowMatrix::getComm() const
    {return J_rowMatrix->getComm();}

    Teuchos::RCP<const NOX::TMap>
    LowRankUpdateRowMatrix::getRowMap() const
    {return J_rowMatrix->getRowMap();}

    Teuchos::RCP<const NOX::TMap>
    LowRankUpdateRowMatrix::getColMap() const
    {return J_rowMatrix->getColMap();}

    Teuchos::RCP<const NOX::TRowGraph>
    LowRankUpdateRowMatrix::getGraph() const
    {return J_rowMatrix->getGraph();}

    
    ::Tpetra::global_size_t LowRankUpdateRowMatrix::getGlobalNumRows() const
    {return J_rowMatrix->getGlobalNumRows();}

    ::Tpetra::global_size_t LowRankUpdateRowMatrix::getGlobalNumCols() const
    {return J_rowMatrix->getGlobalNumCols();}

    size_t LowRankUpdateRowMatrix::getNodeNumRows() const
    {return J_rowMatrix->getNodeNumRows();}

    size_t LowRankUpdateRowMatrix::getNodeNumCols() const
    {return J_rowMatrix->getNodeNumCols();}

    NOX::GlobalOrdinal LowRankUpdateRowMatrix::getIndexBase() const
    {return J_rowMatrix->getIndexBase();}

    ::Tpetra::global_size_t LowRankUpdateRowMatrix::getGlobalNumEntries() const
    {return J_rowMatrix->getGlobalNumEntries();}

    size_t LowRankUpdateRowMatrix::getNodeNumEntries() const
    {return J_rowMatrix->getNodeNumEntries();}

    size_t LowRankUpdateRowMatrix::getNumEntriesInGlobalRow(NOX::GlobalOrdinal globalRow) const
    {return J_rowMatrix->getNumEntriesInGlobalRow(globalRow);}

    size_t LowRankUpdateRowMatrix::getNumEntriesInLocalRow(NOX::LocalOrdinal localRow) const
    {return J_rowMatrix->getNumEntriesInLocalRow(localRow);}

    size_t LowRankUpdateRowMatrix::getGlobalMaxNumRowEntries() const
    {return J_rowMatrix->getGlobalMaxNumRowEntries();}

    size_t LowRankUpdateRowMatrix::getNodeMaxNumRowEntries() const
    {return J_rowMatrix->getNodeMaxNumRowEntries();}

    bool LowRankUpdateRowMatrix::hasColMap() const
    {return J_rowMatrix->hasColMap();}

    bool LowRankUpdateRowMatrix::isLocallyIndexed() const
    {return J_rowMatrix->isLocallyIndexed();}

    bool LowRankUpdateRowMatrix::isGloballyIndexed() const
    {return J_rowMatrix->isGloballyIndexed();}

    bool LowRankUpdateRowMatrix::isFillComplete() const
    {return J_rowMatrix->isFillComplete();}

    bool LowRankUpdateRowMatrix::supportsRowViews() const
    {return J_rowMatrix->supportsRowViews();}

    void
    LowRankUpdateRowMatrix::getGlobalRowCopy(NOX::GlobalOrdinal GlobalRow,
                                             const Teuchos::ArrayView<NOX::GlobalOrdinal> &Indices,
                                             const Teuchos::ArrayView<NOX::Scalar> &Values,
                                             size_t &NumEntries) const
    {}

    void
    LowRankUpdateRowMatrix::getLocalRowCopy (NOX::LocalOrdinal LocalRow,
                                             const Teuchos::ArrayView<NOX::LocalOrdinal> &Indices,
                                             const Teuchos::ArrayView<NOX::Scalar> &Values,
                                             size_t &NumEntries) const
    {}
    
    void
    LowRankUpdateRowMatrix::getGlobalRowView (NOX::GlobalOrdinal GlobalRow,
                                              Teuchos::ArrayView<const NOX::GlobalOrdinal> &indices,
                                              Teuchos::ArrayView<const NOX::Scalar> &values) const
    {}
    
    void
    LowRankUpdateRowMatrix::getLocalRowView(NOX::LocalOrdinal LocalRow,
                                            Teuchos::ArrayView<const NOX::LocalOrdinal>& indices,
                                            Teuchos::ArrayView<const NOX::Scalar>& values) const
    {}

    // ROGER - this is not pure virtual!
    NOX::LocalOrdinal
    LowRankUpdateRowMatrix::getLocalRowViewRaw(const NOX::LocalOrdinal lclRow,
                                               NOX::LocalOrdinal& numEnt,
                                               const NOX::LocalOrdinal*& lclColInds,
                                               const NOX::Scalar*& vals) const
    {}

    void LowRankUpdateRowMatrix::getLocalDiagCopy(NOX::TVector& diag) const
    {}

    void LowRankUpdateRowMatrix::leftScale(const NOX::TVector& x)
    {}
    
    void LowRankUpdateRowMatrix::rightScale(const NOX::TVector& x)
    {}
    
    LowRankUpdateRowMatrix::mag_type LowRankUpdateRowMatrix::getFrobeniusNorm() const
    {}

    // ROGER - this is not pure virtual!
    Teuchos::RCP<NOX::TRowMatrix>
    LowRankUpdateRowMatrix::add(const NOX::Scalar& alpha,
                                 const NOX::TRowMatrix& A,
                                 const NOX::Scalar& beta,
                                 const Teuchos::RCP<const NOX::TMap>& domainMap,
                                 const Teuchos::RCP<const NOX::TMap>& rangeMap,
                                 const Teuchos::RCP<Teuchos::ParameterList>& params) const
    {}

    double LowRankUpdateRowMatrix::computeUV(int MyRow, int MyCol) const
    {}

  } // namespace Tpetra

} // namespace Tpetra

#endif // TPETRA_ROWMATRIX_DECL_HPP
