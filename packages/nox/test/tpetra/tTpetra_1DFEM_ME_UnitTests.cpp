//@HEADER
// ************************************************************************
//
//            NOX: An Object-Oriented Nonlinear Solver Package
//                 Copyright (2002) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Roger Pawlowski (rppawlo@sandia.gov) or
// Eric Phipps (etphipp@sandia.gov), Sandia National Laboratories.
// ************************************************************************
//  CVS Information
//  $Source$
//  $Author$
//  $Date$
//  $Revision$
// ************************************************************************
//@HEADER
#include "Teuchos_ConfigDefs.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_StackedTimer.hpp"

// NOX Objects
#include "NOX.H"
#include "NOX_Thyra.H"

// Trilinos Objects
#include "Teuchos_Comm.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_AbstractFactoryStd.hpp"

#include "Tpetra_Core.hpp"
#include "Tpetra_Vector.hpp"

#include "BelosTypes.hpp"
#include "Stratimikos_DefaultLinearSolverBuilder.hpp"
#include "Thyra_LinearOpWithSolveFactoryHelpers.hpp"
#include "Thyra_Ifpack2PreconditionerFactory.hpp"
#include "Thyra_TpetraThyraWrappers.hpp"
#include "Thyra_VectorStdOps.hpp"
#include "ME_Tpetra_1DFEM.hpp"

#include "NOX_Thyra_MatrixFreeJacobianOperator.hpp"
#include "NOX_MatrixFree_ModelEvaluatorDecorator.hpp"
#include "NOX_TpetraTypedefs.hpp"

TEUCHOS_UNIT_TEST(NOX_Tpetra_1DFEM, Responses)
{
  Teuchos::RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();

  // Get default Tpetra template types
  using Scalar = NOX::Scalar;
  using LO = NOX::LocalOrdinal;
  using GO = NOX::GlobalOrdinal;
  using Node = NOX::NodeType;
  using converter = ::Thyra::TpetraOperatorVectorExtraction<Scalar,LO,GO,Node>;

  // Create the model evaluator object
  Scalar x00 = 0.0;
  Scalar x01 = 1.0;
  const Tpetra::global_size_t numGlobalElements = 100;
  Teuchos::RCP<EvaluatorTpetra1DFEM<Scalar,LO,GO,Node> > model =
    evaluatorTpetra1DFEM<Scalar,LO,GO,Node>(comm, numGlobalElements, x00, x01);

  auto g_thyra = ::Thyra::createMember(*model->get_g_space(4),"g");
  auto DfDp_thyra = Teuchos::rcp_dynamic_cast<::Thyra::MultiVectorBase<Scalar>>(model->create_DfDp_op(2),true);
  auto DgDx_thyra = Teuchos::rcp_dynamic_cast<::Thyra::MultiVectorBase<Scalar>>(model->create_DgDx_op(4),true);
  auto DgDp_thyra = Teuchos::rcp_dynamic_cast<::Thyra::MultiVectorBase<Scalar>>(model->create_DgDp_op(4,2),true);

  auto inArgs = model->createInArgs();
  auto x = ::Thyra::createMember(model->get_x_space(),"x");
  ::Thyra::assign(x.ptr(),1.0);
  inArgs.set_x(x);
  auto outArgs = model->createOutArgs();

  outArgs.set_g(4,::Thyra::ModelEvaluatorBase::Evaluation<::Thyra::VectorBase<Scalar>>(g_thyra));
  outArgs.set_DfDp(2,::Thyra::ModelEvaluatorBase::Derivative<Scalar>(DfDp_thyra));
  outArgs.set_DgDx(4,::Thyra::ModelEvaluatorBase::Derivative<Scalar>(DgDx_thyra));
  outArgs.set_DgDp(4,2,::Thyra::ModelEvaluatorBase::Derivative<Scalar>(DgDp_thyra));

  model->evalModel(inArgs,outArgs);

  TEST_EQUALITY(model->Np(),5);
  TEST_EQUALITY(model->Ng(),5);

  auto g = converter::getTpetraMultiVector(g_thyra);
  auto DfDp = converter::getTpetraMultiVector(DfDp_thyra);
  auto DgDx = converter::getTpetraMultiVector(DgDx_thyra);
  auto DgDp = converter::getTpetraMultiVector(DgDp_thyra);

  // g = T(Zmax) - 2.0
  // x=1.0
  // g= -1.0
  g->sync_host();
  auto g_host = g->getLocalViewHost();
  out << "g = " << g_host(0,0) << std::endl;
  auto tol = std::numeric_limits<Scalar>::epsilon()*100.0;
  TEST_FLOATING_EQUALITY(g_host(0,0),-1.0,tol);

  // DgDx: Right end node is 1, the rest of the vector is zero.
  out << "DgDx:\n";
  DgDx->sync_host();
  DgDx->describe(out,Teuchos::VERB_EXTREME);
  auto DgDx_host = DgDx->getLocalViewHost();
  if (comm->getRank() == (comm->getSize()-1)) {
    TEST_FLOATING_EQUALITY(DgDx_host(DgDx_host.extent(0)-1,0),1.0,tol);
  }
  Teuchos::Array<NOX::TMultiVector::mag_type> norms(1);
  DgDx->norm2(norms);
  TEST_FLOATING_EQUALITY(norms[0],1.0,tol);

  // DfDp
  out << "DfDp:\n";
  DfDp->sync_host();
  DfDp->describe(out,Teuchos::VERB_EXTREME);
  auto DfDp_host = DfDp->getLocalViewHost();
  auto rank = comm->getRank();
  auto size = comm->getSize();
  for (size_t i=0; i < DfDp_host.extent(0); ++i) {
    if ((rank == 0) && (i == 0)) { // left end
      TEST_FLOATING_EQUALITY(DfDp_host(0,0),0.0,tol);
    }
    else if (rank == (size - 1) && (i==(DfDp_host.extent(0)-1))) { // right end
      TEST_FLOATING_EQUALITY(DfDp_host(DfDp_host.extent(0)-1,0),0.005,tol);
    }
    else { // interior nodes
      out << "rank=" << rank << ", i=" << i <<std::endl;
      TEST_FLOATING_EQUALITY(DfDp_host(i,0),0.01,tol);
    }
  }

  out << "DgDp:\n";
  DgDp->sync_host();
  DgDp->describe(out,Teuchos::VERB_EXTREME);
  auto DgDp_host = DgDp->getLocalViewHost();
  TEST_FLOATING_EQUALITY(DgDp_host(0,0),0.0,tol);
}
