// @HEADER
// ****************************************************************************
//                Tempus: Copyright (2017) Sandia Corporation
//
// Distributed under BSD 3-clause license (See accompanying file Copyright.txt)
// ****************************************************************************
// @HEADER

#include "Teuchos_UnitTestHarness.hpp"

#include "Tempus_UnitTest_Utils.hpp"

#include "../TestModels/SinCosModel.hpp"

namespace Tempus_Unit_Test {

using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::rcp_const_cast;
using Teuchos::rcp_dynamic_cast;


// ************************************************************
// ************************************************************
TEUCHOS_UNIT_TEST(ERK_Trapezoidal, Default_Construction)
{
  testExplicitRKAccessorsFullConstruction("RK Explicit Trapezoidal");
}


// ************************************************************
// ************************************************************
TEUCHOS_UNIT_TEST(ERK_Trapezoidal, StepperFactory_Construction)
{
  auto model = rcp(new Tempus_Test::SinCosModel<double>());
  testFactoryConstruction("RK Explicit Trapezoidal", model);
}


// ************************************************************
// ************************************************************
TEUCHOS_UNIT_TEST(ERK_Trapezoidal, AppAction)
{
  testRKAppAction("RK Explicit Trapezoidal", out, success);
}


} // namespace Tempus_Test
