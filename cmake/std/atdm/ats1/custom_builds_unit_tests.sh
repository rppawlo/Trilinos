#!/bin/bash

CURRENT_SCRIPTS_DIR=`echo $BASH_SOURCE | sed "s/\(.*\)\/.*\.sh/\1/g"`
ATDM_CONFIG_SCRIPT_DIR=`readlink -f ${CURRENT_SCRIPTS_DIR}/..`

#
# Test compiler parsing
#

testAll() {

  ATDM_CONFIG_SYSTEM_DIR=${ATDM_CONFIG_SCRIPT_DIR}/ats1

  ATDM_CONFIG_BUILD_NAME=before-intel-19.0.4-mpich-7.7.6_after
  . ${ATDM_CONFIG_SCRIPT_DIR}/utils/set_build_options.sh
  ${_ASSERT_EQUALS_} INTEL-19.0.4_MPICH-7.7.6 ${ATDM_CONFIG_COMPILER}

  ATDM_CONFIG_BUILD_NAME=before-intel-19.0.4_mpich-7.7.6-after
  . ${ATDM_CONFIG_SCRIPT_DIR}/utils/set_build_options.sh
  ${_ASSERT_EQUALS_} INTEL-19.0.4_MPICH-7.7.6 ${ATDM_CONFIG_COMPILER}

  ATDM_CONFIG_BUILD_NAME=before_intel-19.0.4-after
  . ${ATDM_CONFIG_SCRIPT_DIR}/utils/set_build_options.sh
  ${_ASSERT_EQUALS_} INTEL-19.0.4_MPICH-7.7.6 ${ATDM_CONFIG_COMPILER}

  ATDM_CONFIG_BUILD_NAME=before_intel-19-after
  . ${ATDM_CONFIG_SCRIPT_DIR}/utils/set_build_options.sh
  ${_ASSERT_EQUALS_} INTEL-19.0.4_MPICH-7.7.6 ${ATDM_CONFIG_COMPILER}

  ATDM_CONFIG_BUILD_NAME=default
  . ${ATDM_CONFIG_SCRIPT_DIR}/utils/set_build_options.sh
  ${_ASSERT_EQUALS_} INTEL-18.0.5_MPICH-7.7.6 ${ATDM_CONFIG_COMPILER}

  ATDM_CONFIG_BUILD_NAME=before-intel-after
  . ${ATDM_CONFIG_SCRIPT_DIR}/utils/set_build_options.sh
  ${_ASSERT_EQUALS_} INTEL-18.0.5_MPICH-7.7.6 ${ATDM_CONFIG_COMPILER}

  ATDM_CONFIG_BUILD_NAME=before-intel-18.0.5-mpich-7.7.6_after
  . ${ATDM_CONFIG_SCRIPT_DIR}/utils/set_build_options.sh
  ${_ASSERT_EQUALS_} INTEL-18.0.5_MPICH-7.7.6 ${ATDM_CONFIG_COMPILER}

  ATDM_CONFIG_BUILD_NAME=before-intel-18.0.5_mpich-7.7.6-after
  . ${ATDM_CONFIG_SCRIPT_DIR}/utils/set_build_options.sh
  ${_ASSERT_EQUALS_} INTEL-18.0.5_MPICH-7.7.6 ${ATDM_CONFIG_COMPILER}

  ATDM_CONFIG_BUILD_NAME=before_intel-18.0.5-after
  . ${ATDM_CONFIG_SCRIPT_DIR}/utils/set_build_options.sh
  ${_ASSERT_EQUALS_} INTEL-18.0.5_MPICH-7.7.6 ${ATDM_CONFIG_COMPILER}

  ATDM_CONFIG_BUILD_NAME=before_intel-18-after
  . ${ATDM_CONFIG_SCRIPT_DIR}/utils/set_build_options.sh
  ${_ASSERT_EQUALS_} INTEL-18.0.5_MPICH-7.7.6 ${ATDM_CONFIG_COMPILER}

  ATDM_CONFIG_BUILD_NAME=BEFORE-INTEL-AFTER
  . ${ATDM_CONFIG_SCRIPT_DIR}/utils/set_build_options.sh
  ${_ASSERT_EQUALS_} INTEL-18.0.5_MPICH-7.7.6 ${ATDM_CONFIG_COMPILER}

  # Make sure 'arms' does not match 'arm'
  ATDM_CONFIG_BUILD_NAME=anything-intell
  . ${ATDM_CONFIG_SCRIPT_DIR}/utils/set_build_options.sh
  ${_ASSERT_EQUALS_} DEFAULT ${ATDM_CONFIG_COMPILER}

}


#
# Run the unit tests
#

. ${ATDM_CONFIG_SCRIPT_DIR}/test/shunit2/shunit2
