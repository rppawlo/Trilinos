// @HEADER
// *****************************************************************************
//           Panzer: A partial differential equation assembly
//       engine for strongly coupled complex multiphysics systems
//
// Copyright 2011 NTESS and the Panzer contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef PANZER_EVALUATION_POLICY_HPP
#define PANZER_EVALUATION_POLICY_HPP

namespace panzer {

  /**
   * \brief Controls whether a computed field is stored for reuse by later getter calls.
   */
  enum class CacheValues {
    No,  //!< Return the computed field without storing it in the owning object.
    Yes  //!< Store the computed field in the owning object before returning it.
  };

  /**
   * \brief Controls whether a getter recomputes a field even when a cached value is available.
   */
  enum class ForceRecompute {
    No,  //!< Reuse an available cached value.
    Yes  //!< Recompute the requested field.
  };

  /**
   * \brief Controls whether fields computed as dependencies are cached for later reuse.
   */
  enum class CacheDependencies {
    No,  //!< Do not cache fields that are only computed as dependencies.
    Yes  //!< Cache fields that are computed as dependencies.
  };

  /**
   * \brief Controls whether basis values are multiplied by the weighted measure.
   */
  enum class WeightBasis {
    No,  //!< Return unweighted basis values.
    Yes  //!< Return weighted basis values.
  };

} // namespace panzer

#endif
