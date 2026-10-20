// @HEADER
// *****************************************************************************
//        Phalanx: A Partial Differential Equation Field Evaluation
//       Kernel for Flexible Management of Complex Dependency Chains
//
// Copyright 2008 NTESS and the Phalanx contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef PHALANX_TEMPLATE_CONTAINER_FWD_HPP
#define PHALANX_TEMPLATE_CONTAINER_FWD_HPP

namespace PHX {

  // Forward declaration, so that aliases of and pointers to a
  // TemplateContainer can be declared without including its definition.
  // See Phalanx_TemplateContainer.hpp.
  template <typename TypeSeq, template <typename> class ObjectT, typename BaseT = void>
  class TemplateContainer;

}

#endif
