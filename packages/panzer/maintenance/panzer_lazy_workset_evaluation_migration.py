#!/usr/bin/env python3
"""Advisory scanner for Panzer descriptor/cache migration-sensitive code.

This script is intentionally conservative: it reports candidate locations and
suggested review categories. With --apply-patch-sketches, it rewrites only the
safest patch-sketch candidates and leaves manual-review recipes advisory-only.

Common Panzer-application dry-run commands:

  python3 -I panzer_lazy_workset_evaluation_migration.py <application-root> --format text
      Full advisory scan: line findings, migration recipes, and cache hazards.

  python3 -I panzer_lazy_workset_evaluation_migration.py <application-root> --recipes-only
      Human-readable descriptor migration recipes.

  python3 -I panzer_lazy_workset_evaluation_migration.py <application-root> --patch-sketches-only
      Non-applied patch sketches for the safest recipe candidates only.

  python3 -I panzer_lazy_workset_evaluation_migration.py <application-root> --apply-patch-sketches
      Apply the safest descriptor-migration patch sketches in place.

  python3 -I panzer_lazy_workset_evaluation_migration.py <application-root> --cache-hazards-only
      Focused releaseCachedData() lifecycle hazard report.

  python3 -I panzer_lazy_workset_evaluation_migration.py --self-test
      Built-in scanner self-test for lifecycle hazard detection.

Use --format json or --format tsv with the report modes for downstream tooling.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import re
import sys
import tempfile
from collections import Counter, defaultdict
from typing import Iterable

SOURCE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".tpp", ".ipp"
}

SKIP_DIRS = {
    ".git", ".svn", ".hg", "build", "build-trilinos", "cmake-build-debug",
    "cmake-build-release", "CMakeFiles", "Testing", "_deps", "install"
}


@dataclasses.dataclass(frozen=True)
class Rule:
    name: str
    category: str
    risk: str
    pattern: re.Pattern[str]
    hint: str


@dataclasses.dataclass(frozen=True)
class Finding:
    path: pathlib.Path
    line: int
    rule: Rule
    text: str


@dataclasses.dataclass(frozen=True)
class Recipe:
    path: pathlib.Path
    line: int
    name: str
    risk: str
    summary: str
    hint: str
    field_counts: tuple[tuple[str, int], ...] = ()
    phalanx_registration_field_counts: tuple[tuple[str, int], ...] = ()
    legacy_index: str | None = None
    legacy_alias: str | None = None
    setup_function: str | None = None
    access_function: str | None = None
    setup_class: str | None = None
    access_class: str | None = None
    declaration_path: pathlib.Path | None = None
    legacy_member_line: int | None = None
    legacy_member_text: str | None = None
    constructor_function: str | None = None
    constructor_class: str | None = None
    basis_source_line: int | None = None
    basis_source_text: str | None = None
    ir_source_line: int | None = None
    ir_source_text: str | None = None


@dataclasses.dataclass(frozen=True)
class FunctionRegion:
    name: str
    start: int
    end: int
    class_name: str | None = None


@dataclasses.dataclass(frozen=True)
class Assignment:
    line: int
    index: str
    hard_coded_order: bool = False


@dataclasses.dataclass(frozen=True)
class CacheHazard:
    path: pathlib.Path
    line: int
    risk: str
    kind: str
    summary: str
    function: str | None = None
    class_name: str | None = None
    release_line: int | None = None
    handle_line: int | None = None
    handle_name: str | None = None
    text: str | None = None


BASIS_FIELD_GETTERS = {
    "basis_scalar": "getBasisValues(panzer::WeightBasis::No)",
    "basis_vector": "getVectorBasisValues(panzer::WeightBasis::No)",
    "grad_basis": "getGradBasisValues(panzer::WeightBasis::No)",
    "curl_basis_scalar": "getCurlBasisValues(panzer::WeightBasis::No)",
    "curl_basis_vector": "getCurlVectorBasisValues(panzer::WeightBasis::No)",
    "div_basis": "getDivBasisValues(panzer::WeightBasis::No)",
    "weighted_basis_scalar": "getBasisValues(panzer::WeightBasis::Yes)",
    "weighted_basis_vector": "getVectorBasisValues(panzer::WeightBasis::Yes)",
    "weighted_grad_basis": "getGradBasisValues(panzer::WeightBasis::Yes)",
    "weighted_curl_basis_scalar": "getCurlBasisValues(panzer::WeightBasis::Yes)",
    "weighted_curl_basis_vector": "getCurlBasisValues(panzer::WeightBasis::Yes)",
    "weighted_div_basis": "getDivBasisValues(panzer::WeightBasis::Yes)",
    "basis_coordinates_ref": "getBasisCoordinatesRef()",
    "basis_coordinates": "getBasisCoordinates()",
}

BASIS_PHALANX_REGISTRATION_GETTERS = {
    "basis_scalar": "getBasisValuesForPhalanxEvaluatorRegistration()",
    "basis_vector": "getVectorBasisValuesForPhalanxEvaluatorRegistration()",
    "grad_basis": "getGradBasisValuesForPhalanxEvaluatorRegistration()",
    "curl_basis_scalar": "getCurlBasisValuesForPhalanxEvaluatorRegistration()",
    "curl_basis_vector": "getCurlVectorBasisValuesForPhalanxEvaluatorRegistration()",
    "div_basis": "getDivBasisValuesForPhalanxEvaluatorRegistration()",
    "weighted_basis_scalar": "getWeightedBasisValuesForPhalanxEvaluatorRegistration()",
    "weighted_basis_vector": "getWeightedVectorBasisValuesForPhalanxEvaluatorRegistration()",
    "weighted_grad_basis": "getWeightedGradBasisValuesForPhalanxEvaluatorRegistration()",
    "weighted_curl_basis_scalar": "getWeightedCurlBasisValuesForPhalanxEvaluatorRegistration()",
    "weighted_curl_basis_vector": "getWeightedCurlVectorBasisValuesForPhalanxEvaluatorRegistration()",
    "weighted_div_basis": "getWeightedDivBasisValuesForPhalanxEvaluatorRegistration()",
    "basis_coordinates_ref": "getBasisCoordinatesRefForPhalanxEvaluatorRegistration()",
    "basis_coordinates": "getBasisCoordinatesForPhalanxEvaluatorRegistration()",
}

PHALANX_REGISTRATION_CALL_PATTERN = re.compile(
    r"\b(?:addEvaluatedField|addDependentField|addNonConstDependentField|setFieldData)\s*\("
)

INTEGRATION_FIELD_GETTERS = {
    "weighted_measure": "getWeightedMeasure()",
    "jac": "getJacobian()",
    "jac_det": "getJacobianDeterminant()",
    "jac_inv": "getJacobianInverse()",
    "weighted_normals": "getWeightedNormals()",
    "cub_points": "getUniformCubaturePointsRef()",
    "side_cub_points": "manual review: getUniformSideCubaturePointsRef() is shape-preserving but deprecated; getCubaturePointsRef() changes rank/semantics",
    "cub_weights": "getUniformCubatureWeightsRef()",
    "ip_coordinates": "getCubaturePoints()",
    "ref_ip_coordinates": "getCubaturePointsRef()",
    "covarient": "getCovarientMatrix(true)",
    "contravarient": "getContravarientMatrix(true)",
    "norm_contravarient": "getNormContravarientMatrix(true)",
    "surface_normals": "getSurfaceNormals()",
    "surface_rotation_matrices": "getSurfaceRotationMatrices()",
    "node_coordinates": "getNodeCoordinates()",
}

INTEGRATION_FIELD_NOTES = {
    "cub_points": (
        "rank <IP,Dim> uniform reference points. This is the backing field for "
        "getUniformCubaturePointsRef(); for side rules it is mapped into the parent reference cell. "
        "It is not the CV/surface cell-dependent reference-point field."
    ),
    "side_cub_points": (
        "rank <IP,Dim> side-topology reference points. The shape-preserving accessor is the deprecated "
        "getUniformSideCubaturePointsRef(); the preferred getCubaturePointsRef() returns cell-dependent "
        "rank <Cell,IP,Dim> parent-reference points, so automatic replacement needs manual review."
    ),
    "cub_weights": (
        "rank <IP> uniform reference weights. This maps to getUniformCubatureWeightsRef() and is not "
        "the cell-dependent weighted measure."
    ),
    "ip_coordinates": "rank <Cell,IP,Dim> physical integration-point coordinates from getCubaturePoints().",
    "ref_ip_coordinates": "rank <Cell,IP,Dim> cell-dependent reference integration-point coordinates from getCubaturePointsRef().",
}

ALL_LEGACY_FIELDS = tuple(sorted(set(BASIS_FIELD_GETTERS) | set(INTEGRATION_FIELD_GETTERS)))
LEGACY_FIELD_PATTERN = re.compile(
    r"(?:\.|->)\s*(" + "|".join(re.escape(field) for field in ALL_LEGACY_FIELDS) + r")\b"
)
VIEW_HANDLE_ASSIGN_PATTERN = re.compile(
    r"\b(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*[^;]*(?:get_static_view|get_view)\s*\("
)
FIELD_HANDLE_ASSIGN_PATTERN = re.compile(
    r"\b(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*[^;]*\.\s*(?:"
    + "|".join(re.escape(field) for field in ALL_LEGACY_FIELDS)
    + r")\b"
)


FIELD_HANDLE_NAMES = {
    "basis_scalar": "basis_scalar",
    "basis_vector": "basis_values_vector",
    "grad_basis": "grad_basis",
    "curl_basis_scalar": "curl_basis_scalar",
    "curl_basis_vector": "curl_basis_vector",
    "div_basis": "div_basis",
    "weighted_basis_scalar": "weighted_basis",
    "weighted_basis_vector": "weighted_basis_vector",
    "weighted_grad_basis": "weighted_grad_basis",
    "weighted_curl_basis_scalar": "weighted_curl_basis_scalar",
    "weighted_curl_basis_vector": "weighted_curl_basis_vector",
    "weighted_div_basis": "weighted_div_basis",
    "basis_coordinates_ref": "basis_coordinates_ref",
    "basis_coordinates": "basis_coordinates",
    "weighted_measure": "weighted_measure",
    "jac": "jacobian",
    "jac_det": "jacobian_determinant",
    "jac_inv": "jacobian_inverse",
    "weighted_normals": "weighted_normals",
    "cub_points": "cubature_points",
    "side_cub_points": "side_cubature_points",
}


RULES: tuple[Rule, ...] = (
    Rule(
        name="workset-int-rules-private-field",
        category="private-field-breakage",
        risk="high",
        pattern=re.compile(
            r"\(?\s*\b(?:this->wda\([^)]*\)|wda\([^)]*\)|workset)\.int_rules\s*\[[^\]]+\]\s*\)?\s*->\s*"
            r"(?:ip_coordinates|ref_ip_coordinates|weighted_measure|jac|jac_det|jac_inv|"
            r"covarient|contravarient|norm_contravarient|weighted_normals|surface_normals|"
            r"surface_rotation_matrices|cub_points|side_cub_points|cub_weights|node_coordinates)\b"
        ),
        hint="Required fix: direct access to private IntegrationValues2 cached/setup fields must be replaced with IntegrationValues2 getter APIs.",
    ),
    Rule(
        name="workset-bases-private-field",
        category="private-field-breakage",
        risk="high",
        pattern=re.compile(
            r"\(?\s*\b(?:this->wda\([^)]*\)|wda\([^)]*\)|workset)\.bases\s*\[[^\]]+\]\s*\)?\s*->\s*"
            r"(?:basis_scalar|basis_vector|grad_basis|curl_basis_scalar|curl_basis_vector|"
            r"div_basis|weighted_basis_scalar|weighted_basis_vector|weighted_grad_basis|"
            r"weighted_curl_basis_scalar|weighted_curl_basis_vector|weighted_div_basis|"
            r"basis_coordinates_ref|basis_coordinates)\b"
        ),
        hint="Required fix: direct access to private BasisValues2 cached/setup fields must be replaced with value getters or Phalanx evaluator registration getters as appropriate.",
    ),
    Rule(
        name="workset-int-rules-index",
        category="descriptor-access",
        risk="medium",
        pattern=re.compile(r"\b(?:this->wda\([^)]*\)|wda\([^)]*\)|workset)\.int_rules\s*\["),
        hint="Recommended descriptor migration: direct workset int_rules indexing may still compile when using getters, but prefer WorksetDetails::getIntegrationValues(IntegrationDescriptor, true).",
    ),
    Rule(
        name="workset-bases-index",
        category="descriptor-access",
        risk="medium",
        pattern=re.compile(r"\b(?:this->wda\([^)]*\)|wda\([^)]*\)|workset)\.bases\s*\["),
        hint="Recommended descriptor migration: direct workset bases indexing may still compile when using getters, but prefer WorksetDetails::getBasisValues(BasisDescriptor, IntegrationDescriptor, true).",
    ),
    Rule(
        name="integration-rule-index-helper",
        category="descriptor-access",
        risk="high",
        pattern=re.compile(r"\bgetIntegrationRuleIndex\s*\("),
        hint="Store an IntegrationDescriptor and reacquire with getIntegrationValues(...).",
    ),
    Rule(
        name="basis-index-helper",
        category="descriptor-access",
        risk="high",
        pattern=re.compile(r"\bgetBasisIndex\s*\("),
        hint="Store a BasisDescriptor/IntegrationDescriptor and reacquire with getBasisValues(...).",
    ),
    Rule(
        name="integration-values-direct-field",
        category="direct-field-access",
        risk="medium",
        pattern=re.compile(
            r"(?:\(?\s*\b(?:this->wda\([^)]*\)|wda\([^)]*\)|workset)\.int_rules\s*\[[^\]]+\]\s*\)?"
            r"|\b(?:[A-Za-z_]\w*(?:integration|Integration|values|Values|int_rule|ir|IR)[A-Za-z0-9_]*|iv)\b)"
            r"\s*(?:\.|->)\s*(?:ip_coordinates|ref_ip_coordinates|weighted_measure|jac|jac_det|jac_inv|"
            r"covarient|contravarient|norm_contravarient|weighted_normals|surface_normals|"
            r"surface_rotation_matrices|cub_points|side_cub_points|cub_weights|node_coordinates)\b"
        ),
        hint="Prefer IntegrationValues2 getter APIs near point-of-use; review retained handles carefully.",
    ),
    Rule(
        name="basis-values-direct-field",
        category="direct-field-access",
        risk="medium",
        pattern=re.compile(
            r"(?:\(?\s*\b(?:this->wda\([^)]*\)|wda\([^)]*\)|workset)\.bases\s*\[[^\]]+\]\s*\)?"
            r"|\b(?:[A-Za-z_]\w*(?:basis|Basis|values|Values)[A-Za-z0-9_]*|bv)\b)"
            r"\s*(?:\.|->)\s*(?:basis_scalar|basis_vector|grad_basis|curl_basis_scalar|curl_basis_vector|"
            r"div_basis|weighted_basis_scalar|weighted_basis_vector|weighted_grad_basis|"
            r"weighted_curl_basis_scalar|weighted_curl_basis_vector|weighted_div_basis|"
            r"basis_coordinates_ref|basis_coordinates)\b"
        ),
        hint="Prefer BasisValues2 getter APIs near point-of-use; review retained handles carefully.",
    ),
    Rule(
        name="integration-getter-nonconst-mdfield-ref",
        category="const-getter-fallout",
        risk="medium",
        pattern=re.compile(
            r"\bPHX::MDField\s*<\s*(?!const\b)[^,>]+,\s*[^>]*\b(?:panzer::)?IP\b[^>]*>\s*(?:const\s*)?&"
        ),
        hint=(
            "IntegrationValues2 getters return const-valued MDFields. If this field is populated from an "
            "IntegrationValues2 getter, change the parameter/helper type to PHX::MDField<const Scalar,...>; "
            "leave mutable output fields unchanged."
        ),
    ),
    Rule(
        name="basis-getter-bool-first-arg",
        category="bool-api",
        risk="medium",
        pattern=re.compile(
            r"\bget(?:BasisValues|VectorBasisValues|GradBasisValues|Curl2DVectorBasis|"
            r"CurlVectorBasis|DivVectorBasis)\s*\(\s*(?:true|false)\b"
        ),
        hint="First bool is weighting. Consider WeightBasis::{Yes,No}, CacheValues, and ForceRecompute for clarity.",
    ),
    Rule(
        name="cache-getter-bool-first-arg",
        category="bool-api",
        risk="low",
        pattern=re.compile(
            r"\bget(?:Jacobian|JacobianInverse|JacobianDeterminant|WeightedMeasure|WeightedNormals|"
            r"CubaturePoints|CubaturePointsRef|BasisCoordinatesRef|BasisCoordinates|"
            r"BasisValuesRef|VectorBasisValuesRef|GradBasisValuesRef|Curl2DVectorBasisRef|"
            r"CurlVectorBasisRef|DivVectorBasisRef)\s*\(\s*(?:true|false)\b"
        ),
        hint="Bool is a cache control. Consider CacheValues::{Yes,No} and ForceRecompute for clarity.",
    ),
    Rule(
        name="release-cached-data",
        category="lifecycle",
        risk="high",
        pattern=re.compile(r"\breleaseCachedData\s*\("),
        hint="After releaseCachedData(), reacquire fields/views from getters before use.",
    ),
    Rule(
        name="reset-arrays",
        category="lifecycle",
        risk="medium",
        pattern=re.compile(r"\bresetArrays\s*\("),
        hint="resetArrays() is logical invalidation, not guaranteed memory release.",
    ),
    Rule(
        name="temporary-getter-view-hazard",
        category="lifecycle",
        risk="medium",
        pattern=re.compile(
            r"\b(?:Kokkos::subview|Kokkos::create_mirror_view(?:_and_copy)?|Kokkos::deep_copy)\s*\([^;]*"
            r"\.\s*get(?:Jacobian|JacobianInverse|JacobianDeterminant|WeightedMeasure|WeightedNormals|"
            r"CubaturePoints|CubaturePointsRef|UniformCubaturePointsRef|UniformCubatureWeightsRef|"
            r"BasisValues|VectorBasisValues|GradBasisValues|CurlBasisValues|CurlVectorBasisValues|"
            r"DivBasisValues|BasisCoordinatesRef|BasisCoordinates)\s*\([^;]*\)\s*\.\s*(?:get_static_view|get_view)\s*\("
        ),
        hint=(
            "Do not take a view/subview directly from a temporary lazy getter result. Store the getter result "
            "in a named local first, then take get_view()/get_static_view() from that local."
        ),
    ),
    Rule(
        name="stored-view-handle",
        category="lifecycle",
        risk="high",
        pattern=re.compile(r"\b(?:auto|const\s+auto|typename\s+[^=;]+|PHX::View<[^>]+>)\s+\w+\s*=\s*[^;]*(?:get_static_view|get_view)\s*\("),
        hint="Review whether this view is retained across phases or cache release; reacquire near use if needed.",
    ),
)


def iter_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in SOURCE_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def scan_file(path: pathlib.Path) -> list[Finding]:
    findings: list[Finding] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"warning: could not read {path}: {exc}", file=sys.stderr)
        return findings

    in_block_comment = False
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if in_block_comment:
            if "*/" in stripped:
                in_block_comment = False
                stripped = stripped.split("*/", 1)[1].strip()
            else:
                continue
        if stripped.startswith("/*"):
            if "*/" in stripped:
                stripped = stripped.split("*/", 1)[1].strip()
            else:
                in_block_comment = True
                continue
        if stripped.startswith("//"):
            continue
        for rule in RULES:
            if rule.pattern.search(stripped):
                findings.append(Finding(path=path, line=line_no, rule=rule, text=stripped))
    return findings


def first_matching_line(lines: list[str], pattern: re.Pattern[str]) -> int | None:
    for line_no, line in enumerate(lines, start=1):
        if pattern.search(line):
            return line_no
    return None


def function_context_from_signature(signature: str) -> tuple[str, str | None] | None:
    compact = " ".join(signature.split())
    header = compact.split("(", 1)[0]
    if "::" not in header:
        return None
    qualifier, tail = header.rsplit("::", 1)
    function_match = re.search(r"~?\w+\s*$", tail)
    if not function_match:
        return None

    class_name = None
    class_match = re.search(r"([A-Za-z_]\w*)\s*(?:<[^;{}]*>)?\s*$", qualifier)
    if class_match:
        class_name = class_match.group(1)

    return function_match.group(0).strip(), class_name


def find_function_regions(lines: list[str]) -> list[FunctionRegion]:
    regions: list[FunctionRegion] = []
    pending_start: int | None = None
    pending_lines: list[str] = []
    brace_depth = 0
    active_start: int | None = None
    active_name: str | None = None
    active_class_name: str | None = None

    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if active_start is None:
            if pending_start is None:
                control_statement = re.match(r"(?:for|if|while|switch)\s*\(", stripped)
                scope_operator = line.find("::")
                first_paren = line.find("(")
                looks_like_signature = (
                    scope_operator >= 0
                    and not stripped.startswith("//")
                    and not stripped.endswith(";")
                    and not control_statement
                    and (first_paren < 0 or scope_operator < first_paren)
                )
                if looks_like_signature:
                    pending_start = line_no
                    pending_lines = [line]
            else:
                pending_lines.append(line)

            if pending_start is not None and "{" in line:
                signature = "\n".join(pending_lines)
                context = function_context_from_signature(signature)
                if context:
                    name, class_name = context
                    active_start = pending_start
                    active_name = name
                    active_class_name = class_name
                    brace_depth = line.count("{") - line.count("}")
                    if brace_depth <= 0:
                        regions.append(FunctionRegion(active_name, active_start, line_no, active_class_name))
                        active_start = None
                        active_name = None
                        active_class_name = None
                        brace_depth = 0
                    pending_start = None
                    pending_lines = []
                else:
                    pending_start = None
                    pending_lines = []
            elif pending_start is not None and line_no - pending_start > 8:
                pending_start = None
                pending_lines = []
        else:
            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0:
                regions.append(FunctionRegion(active_name or "<unknown>", active_start, line_no, active_class_name))
                active_start = None
                active_name = None
                active_class_name = None
                brace_depth = 0

    return regions


def region_containing(regions: list[FunctionRegion], line: int | None) -> FunctionRegion | None:
    if line is None:
        return None
    for region in regions:
        if region.start <= line <= region.end:
            return region
    return None


def region_text(lines: list[str], region: FunctionRegion | None) -> str:
    if region is None:
        return "\n".join(lines)
    return "\n".join(lines[region.start - 1:region.end])


def scan_cache_hazards_from_lines(path: pathlib.Path, lines: list[str]) -> list[CacheHazard]:
    hazards: list[CacheHazard] = []
    regions = find_function_regions(lines)
    for region in regions:
        region_lines = list(enumerate(lines[region.start - 1:region.end], start=region.start))
        release_lines = [
            (line_no, line.strip())
            for line_no, line in region_lines
            if re.search(r"\breleaseCachedData\s*\(", line)
        ]
        if not release_lines:
            continue

        handles: dict[str, tuple[int, str]] = {}
        for line_no, line in region_lines:
            view_match = VIEW_HANDLE_ASSIGN_PATTERN.search(line)
            field_match = FIELD_HANDLE_ASSIGN_PATTERN.search(line)
            match = view_match or field_match
            if match:
                handles[match.group("name")] = (line_no, line.strip())

        for release_line, _release_text in release_lines:
            for line_no, line in region_lines:
                if line_no <= release_line:
                    continue
                stripped = line.strip()
                if LEGACY_FIELD_PATTERN.search(line):
                    hazards.append(CacheHazard(
                        path=path,
                        line=line_no,
                        risk="high",
                        kind="direct-field-after-release",
                        summary="Direct BasisValues2/IntegrationValues2 field access occurs after releaseCachedData() in the same function.",
                        function=region.name,
                        class_name=region.class_name,
                        release_line=release_line,
                        text=stripped,
                    ))
                if re.search(r"\b(?:get_static_view|get_view)\s*\(", line):
                    hazards.append(CacheHazard(
                        path=path,
                        line=line_no,
                        risk="medium",
                        kind="view-after-release",
                        summary="A view handle is acquired after releaseCachedData(); verify the owning field was reacquired through a getter first.",
                        function=region.name,
                        class_name=region.class_name,
                        release_line=release_line,
                        text=stripped,
                    ))
                for name, (handle_line, handle_text) in handles.items():
                    if handle_line >= release_line:
                        continue
                    if re.search(r"\b" + re.escape(name) + r"\b", line):
                        hazards.append(CacheHazard(
                            path=path,
                            line=line_no,
                            risk="high",
                            kind="retained-handle-after-release",
                            summary="A handle assigned before releaseCachedData() appears to be used after the release in the same function.",
                            function=region.name,
                            class_name=region.class_name,
                            release_line=release_line,
                            handle_line=handle_line,
                            handle_name=name,
                            text=f"handle from line {handle_line}: {handle_text}; later use: {stripped}",
                        ))

    unique: dict[tuple[str, int, str, int | None, str | None], CacheHazard] = {}
    for hazard in hazards:
        key = (str(hazard.path), hazard.line, hazard.kind, hazard.release_line, hazard.handle_name)
        unique[key] = hazard
    return sorted(unique.values(), key=lambda h: (str(h.path), h.line, h.kind))


def scan_cache_hazards(path: pathlib.Path) -> list[CacheHazard]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        print(f"warning: could not read {path}: {exc}", file=sys.stderr)
        return []
    return scan_cache_hazards_from_lines(path, lines)


def count_direct_fields(text: str, field_getters: dict[str, str]) -> tuple[tuple[str, int], ...]:
    counts: Counter[str] = Counter()
    pattern = re.compile(r"(?:\.|->)\s*(" + "|".join(re.escape(field) for field in field_getters) + r")\b")
    for match in pattern.finditer(text):
        counts[match.group(1)] += 1
    return tuple(sorted(counts.items()))


def count_alias_fields(text: str, alias: str, field_getters: dict[str, str]) -> tuple[tuple[str, int], ...]:
    counts: Counter[str] = Counter()
    pattern = re.compile(
        r"\b" + re.escape(alias) + r"\s*(?:\.|->)\s*(" +
        "|".join(re.escape(field) for field in field_getters) + r")\b"
    )
    for match in pattern.finditer(text):
        counts[match.group(1)] += 1
    return tuple(sorted(counts.items()))


def count_indexed_fields(text: str, container: str, index: str, field_getters: dict[str, str]) -> tuple[tuple[str, int], ...]:
    counts: Counter[str] = Counter()
    pattern = re.compile(
        r"\bworkset\." + re.escape(container) + r"\s*\[\s*" + index_pattern(index) +
        r"\s*\]\s*->\s*(" + "|".join(re.escape(field) for field in field_getters) + r")\b"
    )
    for match in pattern.finditer(text):
        counts[match.group(1)] += 1
    return tuple(sorted(counts.items()))


def count_phalanx_registration_fields(text: str, field_getters: dict[str, str], container: str,
                                      index: str | None, alias: str | None) -> tuple[tuple[str, int], ...]:
    counts: Counter[str] = Counter()
    fields = "|".join(re.escape(field) for field in field_getters)
    direct_pattern = re.compile(r"(?:\.|->)\s*(" + fields + r")\b")
    alias_pattern = re.compile(
        r"\b" + re.escape(alias) + r"\s*(?:\.|->)\s*(" + fields + r")\b"
    ) if alias else None
    indexed_pattern = re.compile(
        r"\bworkset\." + re.escape(container) + r"\s*\[\s*" + index_pattern(index) +
        r"\s*\]\s*->\s*(" + fields + r")\b"
    ) if index else None

    for line in text.splitlines():
        if not PHALANX_REGISTRATION_CALL_PATTERN.search(line):
            continue
        scoped_counted = False
        if alias_pattern:
            for match in alias_pattern.finditer(line):
                counts[match.group(1)] += 1
                scoped_counted = True
        if indexed_pattern:
            for match in indexed_pattern.finditer(line):
                counts[match.group(1)] += 1
                scoped_counted = True
        if scoped_counted:
            continue
        for match in direct_pattern.finditer(line):
            counts[match.group(1)] += 1
    return tuple(sorted(counts.items()))


def merge_field_counts(*field_counts: tuple[tuple[str, int], ...]) -> tuple[tuple[str, int], ...]:
    counts: Counter[str] = Counter()
    for items in field_counts:
        counts.update(dict(items))
    return tuple(sorted(counts.items()))


def find_assignments(lines: list[str], helper: str) -> list[Assignment]:
    pattern = re.compile(
        r"\b(?P<index>(?:this->)?\w+)\s*=\s*panzer::" + re.escape(helper) + r"\s*\("
    )
    assignments: list[Assignment] = []
    for line_no, line in enumerate(lines, start=1):
        match = pattern.search(line)
        if match:
            hard_coded_order = False
            if helper == "getIntegrationRuleIndex":
                hard_coded_order = bool(re.search(r"\bgetIntegrationRuleIndex\s*\(\s*\d+\s*,", line))
            assignments.append(Assignment(line_no, match.group("index"), hard_coded_order))
    return assignments


def find_assignment(lines: list[str], helper: str) -> tuple[int | None, str | None, bool]:
    assignments = find_assignments(lines, helper)
    if not assignments:
        return None, None, False
    assignment = assignments[0]
    return assignment.line, assignment.index, assignment.hard_coded_order


def index_pattern(index: str) -> str:
    if index.startswith("this->"):
        return r"(?:this->)?" + re.escape(index.removeprefix("this->"))
    return re.escape(index)


def find_values_alias(lines: list[str], container: str, index: str | None) -> tuple[int | None, str | None]:
    if index is None:
        return None, None
    pattern = re.compile(
        r"\b(?P<alias>\w+)\s*=\s*\*?\s*\(?\s*workset\." + re.escape(container) +
        r"\s*\[\s*" + index_pattern(index) + r"\s*\](?!\s*->)"
    )
    for line_no, line in enumerate(lines, start=1):
        match = pattern.search(line)
        if match:
            return line_no, match.group("alias")
    return None, None


def first_index_access_line(lines: list[str], container: str, index: str | None) -> int | None:
    if index is None:
        return None
    pattern = re.compile(
        r"(?:(?:this->)?wda\([^)]*\)|workset)\." + re.escape(container) +
        r"\s*\[\s*" + index_pattern(index) + r"\s*\]"
    )
    return first_matching_line(lines, pattern)


def field_counts_for_pattern(text: str, field_getters: dict[str, str], container: str,
                             index: str | None, alias: str | None) -> tuple[tuple[str, int], ...]:
    if alias:
        alias_counts = count_alias_fields(text, alias, field_getters)
    else:
        alias_counts = ()
    if index:
        indexed_counts = count_indexed_fields(text, container, index, field_getters)
    else:
        indexed_counts = ()
    scoped_counts = merge_field_counts(alias_counts, indexed_counts)
    return scoped_counts if scoped_counts else count_direct_fields(text, field_getters)


def candidate_declaration_path(path: pathlib.Path) -> pathlib.Path | None:
    name = path.name
    candidates: list[pathlib.Path] = []
    if name.endswith("_impl.hpp"):
        stem = name.removesuffix("_impl.hpp")
        candidates.append(path.with_name(stem + ".hpp"))
        candidates.append(path.with_name(stem + "_decl.hpp"))
    if name.endswith("_impl.h"):
        stem = name.removesuffix("_impl.h")
        candidates.append(path.with_name(stem + ".h"))
        candidates.append(path.with_name(stem + "_decl.h"))
    if path.suffix in {".cpp", ".cc", ".cxx"}:
        for suffix in (".hpp", ".h", ".hh"):
            candidates.append(path.with_suffix(suffix))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_legacy_member(path: pathlib.Path | None, index: str | None) -> tuple[int | None, str | None]:
    if path is None or index is None:
        return None, None
    member_name = index.removeprefix("this->")
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None, None

    pattern = re.compile(r"\b" + re.escape(member_name) + r"\b")
    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        if pattern.search(line) and ";" in line:
            return line_no, stripped
    return None, None


def source_line_score(line: str) -> int:
    score = 0
    if '"Basis"' in line or '"IR"' in line:
        score += 8
    if ".get<" in line or "get<" in line:
        score += 4
    if "valid_params" in line or ".set<" in line:
        score += 2
    if "using " in line:
        score -= 4
    return score


def find_best_source_line(lines: list[str], region: FunctionRegion, pattern: re.Pattern[str]) -> tuple[int | None, str | None]:
    matches: list[tuple[int, int, str]] = []
    for line_no in range(region.start, region.end + 1):
        line = lines[line_no - 1]
        if pattern.search(line):
            matches.append((source_line_score(line), line_no, line.strip()))
    if not matches:
        return None, None
    _score, line_no, text = max(matches, key=lambda item: (item[0], -item[1]))
    return line_no, text


def find_constructor_context(lines: list[str], regions: list[FunctionRegion], class_name: str | None) -> dict[str, object]:
    context: dict[str, object] = {
        "constructor_function": None,
        "constructor_class": None,
        "basis_source_line": None,
        "basis_source_text": None,
        "ir_source_line": None,
        "ir_source_text": None,
    }
    if class_name is None:
        return context

    for region in regions:
        if region.class_name != class_name or region.name != class_name:
            continue
        basis_line, basis_text = find_best_source_line(
            lines, region, re.compile(r'"Basis"|\bBasisIRLayout\b|\bPureBasis\b')
        )
        ir_line, ir_text = find_best_source_line(
            lines, region, re.compile(r'"IR"|\bIntegrationRule\b')
        )
        if basis_line is not None or ir_line is not None:
            context.update({
                "constructor_function": region.name,
                "constructor_class": region.class_name,
                "basis_source_line": basis_line,
                "basis_source_text": basis_text,
                "ir_source_line": ir_line,
                "ir_source_text": ir_text,
            })
            return context
    return context


def field_getter(field: str, *, phalanx_registration: bool = False) -> str:
    if phalanx_registration and field in BASIS_PHALANX_REGISTRATION_GETTERS:
        return BASIS_PHALANX_REGISTRATION_GETTERS[field]
    if field in BASIS_FIELD_GETTERS:
        return BASIS_FIELD_GETTERS[field]
    return INTEGRATION_FIELD_GETTERS[field]


def count_dict(field_counts: tuple[tuple[str, int], ...]) -> dict[str, int]:
    return dict(field_counts)


def format_field_mappings(field_counts: tuple[tuple[str, int], ...],
                          phalanx_registration_field_counts: tuple[tuple[str, int], ...] = ()) -> list[str]:
    lines: list[str] = []
    registration_counts = count_dict(phalanx_registration_field_counts)
    for field, count in field_counts:
        registration_count = registration_counts.get(field, 0)
        note = INTEGRATION_FIELD_NOTES.get(field)
        note_text = f" ({note})" if note else ""
        if registration_count == count and field in BASIS_PHALANX_REGISTRATION_GETTERS:
            lines.append(f"      {field}: {count} Phalanx registration use(s) -> {field_getter(field, phalanx_registration=True)}")
        elif registration_count:
            lines.append(
                f"      {field}: {count} use(s), including {registration_count} Phalanx registration use(s) -> manual review"
            )
        else:
            lines.append(f"      {field}: {count} use(s) -> {field_getter(field)}{note_text}")
    return lines


def suggested_owner_handle(recipe: Recipe) -> str | None:
    if recipe.name.startswith("basis-index"):
        return "const auto& basis_values = workset.getBasisValues(basis_descriptor, integration_descriptor, false);"
    if recipe.name.startswith("integration-index"):
        return "const auto& integration_values = workset.getIntegrationValues(integration_descriptor, false);"
    return None


def suggested_field_handles(recipe: Recipe) -> list[dict[str, object]]:
    owner = "basis_values" if recipe.name.startswith("basis-index") else "integration_values"
    registration_counts = count_dict(recipe.phalanx_registration_field_counts)
    suggestions: list[dict[str, object]] = []
    for field, count in recipe.field_counts:
        registration_count = registration_counts.get(field, 0)
        phalanx_registration = (
            recipe.name.startswith("basis-index")
            and field in BASIS_PHALANX_REGISTRATION_GETTERS
            and registration_count == count
        )
        mixed_registration_context = 0 < registration_count < count
        getter = field_getter(field, phalanx_registration=phalanx_registration)
        manual_review = getter.startswith("manual review") or mixed_registration_context
        handle_name = FIELD_HANDLE_NAMES.get(field, field)
        declaration = None
        if not manual_review and " or manual " not in getter:
            qualifier = "auto" if phalanx_registration else "const auto"
            declaration = f"{qualifier} {handle_name} = {owner}.{getter};"
        suggestions.append({
            "field": field,
            "count": count,
            "phalanx_registration_count": registration_count,
            "phalanx_registration": phalanx_registration,
            "mixed_registration_context": mixed_registration_context,
            "handle": handle_name,
            "getter": getter,
            "declaration": declaration,
            "manual_review": manual_review or "manual" in getter,
        })
    return suggestions


def legacy_field_expression(recipe: Recipe, field: str) -> str:
    if recipe.legacy_alias:
        return f"{recipe.legacy_alias}.{field}"
    if recipe.name.startswith("basis-index"):
        return f"workset.bases[{recipe.legacy_index}]->{field}"
    return f"workset.int_rules[{recipe.legacy_index}]->{field}"


def suggested_replacements(recipe: Recipe) -> list[dict[str, object]]:
    replacements: list[dict[str, object]] = []
    for suggestion in suggested_field_handles(recipe):
        field = str(suggestion["field"])
        handle = str(suggestion["handle"])
        manual_review = bool(suggestion["manual_review"])
        replacement = None if manual_review else handle
        replacements.append({
            "field": field,
            "legacy": legacy_field_expression(recipe, field),
            "replacement": replacement,
            "manual_review": manual_review,
        })
    return replacements


def format_replacements(recipe: Recipe) -> list[str]:
    lines: list[str] = []
    for replacement in suggested_replacements(recipe):
        if replacement["manual_review"]:
            lines.append(f"      {replacement['legacy']} -> manual review")
        else:
            lines.append(f"      {replacement['legacy']} -> {replacement['replacement']}")
    return lines


def source_variable_name(source_text: str | None) -> str | None:
    if source_text is None:
        return None
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*=", source_text)
    if match:
        return match.group(1)
    # Constructor arguments such as `const panzer::IntegrationRule & ir` are
    # common in Panzer applications and are enough to build descriptors.
    match = re.search(
        r"\bpanzer::(?:IntegrationRule|PureBasis|BasisIRLayout)\s*(?:const\s*)?(?:[&*]\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*(?:[,;)])",
        source_text,
    )
    if match:
        return match.group(1)
    match = re.search(
        r"\bTeuchos::RCP\s*<\s*(?:const\s+)?panzer::(?:IntegrationRule|PureBasis|BasisIRLayout)\s*>\s*(?:const\s*)?(?:[&*]\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*(?:[,;)])",
        source_text,
    )
    if match:
        return match.group(1)
    return None


def source_uses_pointer(source_text: str | None) -> bool:
    if source_text is None:
        return True
    if "Teuchos::RCP" in source_text or "->" in source_text:
        return True
    return "*" in source_text and "&" not in source_text


def basis_source_kind(source_text: str | None) -> str | None:
    if source_text is None:
        return None
    if "BasisIRLayout" in source_text:
        return "BasisIRLayout"
    if "PureBasis" in source_text:
        return "PureBasis"
    if '"Basis"' in source_text:
        return "Basis parameter"
    return None


def integration_source_kind(source_text: str | None) -> str | None:
    if source_text is None:
        return None
    if "IntegrationRule" in source_text:
        return "IntegrationRule"
    if '"IR"' in source_text:
        return "IR parameter"
    return None


def suggested_descriptor_initializers(recipe: Recipe) -> list[dict[str, object]]:
    suggestions: list[dict[str, object]] = []

    if recipe.name.startswith("basis-index"):
        basis_kind = basis_source_kind(recipe.basis_source_text)
        basis_variable = source_variable_name(recipe.basis_source_text)
        basis_expression: str
        basis_manual = False
        if basis_kind == "BasisIRLayout" and basis_variable:
            accessor = "->" if source_uses_pointer(recipe.basis_source_text) else "."
            basis_expression = (
                f"basis_descriptor = panzer::BasisDescriptor({basis_variable}{accessor}getBasis()->order(), "
                f"{basis_variable}{accessor}getBasis()->type());"
            )
        elif basis_kind == "PureBasis" and basis_variable:
            accessor = "->" if source_uses_pointer(recipe.basis_source_text) else "."
            basis_expression = (
                f"basis_descriptor = panzer::BasisDescriptor({basis_variable}{accessor}order(), "
                f"{basis_variable}{accessor}type());"
            )
        else:
            basis_expression = "basis_descriptor = panzer::BasisDescriptor(<basis_order>, <basis_type>);"
            basis_manual = True
        suggestions.append({
            "descriptor": "basis_descriptor",
            "source_kind": basis_kind,
            "source_variable": basis_variable,
            "expression": basis_expression,
            "manual_review": basis_manual,
        })

    if recipe.name.startswith("basis-index") or recipe.name.startswith("integration-index"):
        ir_kind = integration_source_kind(recipe.ir_source_text)
        ir_variable = source_variable_name(recipe.ir_source_text)
        ir_expression: str
        ir_manual = False
        if ir_kind == "IntegrationRule" and ir_variable:
            accessor = "->" if source_uses_pointer(recipe.ir_source_text) else "."
            ir_expression = (
                f"integration_descriptor = panzer::IntegrationDescriptor({ir_variable}{accessor}getOrder(), "
                f"{ir_variable}{accessor}getType(), {ir_variable}{accessor}getSide());"
            )
        else:
            ir_expression = "integration_descriptor = panzer::IntegrationDescriptor(<cubature_order>, <integration_type>, <side>);"
            ir_manual = True
        suggestions.append({
            "descriptor": "integration_descriptor",
            "source_kind": ir_kind,
            "source_variable": ir_variable,
            "expression": ir_expression,
            "manual_review": ir_manual,
        })

    return suggestions


def format_descriptor_initializers(recipe: Recipe) -> list[str]:
    lines: list[str] = []
    for suggestion in suggested_descriptor_initializers(recipe):
        source_kind = suggestion["source_kind"] or "unknown source"
        source_variable = suggestion["source_variable"] or "<unknown>"
        review_marker = " // manual review" if suggestion["manual_review"] else ""
        lines.append(
            f"      // from {source_kind} `{source_variable}`"
        )
        lines.append(f"      {suggestion['expression']}{review_marker}")
    return lines


def declaration_header_text(path: pathlib.Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def include_insertion_point(path: pathlib.Path | None) -> dict[str, object] | None:
    text = declaration_header_text(path)
    if not text:
        return None

    lines = text.splitlines()
    panzer_includes: list[tuple[int, str]] = []
    includes: list[tuple[int, str]] = []
    guard_lines: list[tuple[int, str]] = []
    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#include"):
            includes.append((line_no, stripped))
            if "Panzer_" in stripped:
                panzer_includes.append((line_no, stripped))
        if stripped.startswith("#define "):
            guard_lines.append((line_no, stripped))

    if panzer_includes:
        line_no, text_line = panzer_includes[-1]
        return {
            "line": line_no,
            "text": text_line,
            "strategy": "after last existing Panzer include",
        }
    if includes:
        line_no, text_line = includes[-1]
        return {
            "line": line_no,
            "text": text_line,
            "strategy": "after last existing include",
        }
    if guard_lines:
        line_no, text_line = guard_lines[-1]
        return {
            "line": line_no,
            "text": text_line,
            "strategy": "after include guard define",
        }
    return {
        "line": 1,
        "text": lines[0].strip() if lines else "",
        "strategy": "near top of declaration header",
    }


def suggested_include_changes(recipe: Recipe) -> list[dict[str, object]]:
    if recipe.declaration_path is None:
        return []

    required_headers: list[tuple[str, str]] = []
    if recipe.name.startswith("basis-index"):
        required_headers.append(("BasisDescriptor", '#include "Panzer_BasisDescriptor.hpp"'))
    if recipe.name.startswith("basis-index") or recipe.name.startswith("integration-index"):
        required_headers.append(("IntegrationDescriptor", '#include "Panzer_IntegrationDescriptor.hpp"'))

    text = declaration_header_text(recipe.declaration_path)
    changes: list[dict[str, object]] = []
    for descriptor, include_line in required_headers:
        include_present = include_line in text
        forward_declared = f"class {descriptor}" in text or f"struct {descriptor}" in text
        if include_present:
            action = "include already present"
            needed = False
        elif forward_declared:
            action = "add include; forward declaration is not enough for by-value descriptor member"
            needed = True
        else:
            action = "add include to declaration header"
            needed = True
        changes.append({
            "descriptor": descriptor,
            "include": include_line,
            "include_present": include_present,
            "forward_declared": forward_declared,
            "needed": needed,
            "action": action,
        })
    return changes


def format_include_insertion_point(recipe: Recipe) -> list[str]:
    point = include_insertion_point(recipe.declaration_path)
    if point is None:
        return []
    return [
        f"      insertion point: place near line {point['line']} ({point['strategy']}): {point['text']}"
    ]


def format_include_changes(recipe: Recipe) -> list[str]:
    lines: list[str] = []
    lines.extend(format_include_insertion_point(recipe))
    for change in suggested_include_changes(recipe):
        if change["needed"]:
            lines.append(f"      {change['include']}  // {change['action']}")
        else:
            lines.append(f"      // {change['descriptor']}: {change['action']}")
    return lines


def recipe_classification(recipe: Recipe) -> tuple[str, tuple[str, ...]]:
    reasons: list[str] = []
    manual_fields = [
        suggestion["field"]
        for suggestion in suggested_field_handles(recipe)
        if suggestion["manual_review"]
    ]

    if "missing-ir" in recipe.name:
        reasons.append("no obvious local integration descriptor source")
    if "hard-coded-order" in recipe.name:
        reasons.append("hard-coded integration order must be matched to a descriptor source")
    if manual_fields:
        reasons.append("manual-review field mapping: " + ", ".join(str(field) for field in manual_fields))
    if recipe.setup_function is None:
        reasons.append("setup function was not identified")
    if recipe.access_function is None:
        reasons.append("access function was not identified")
    if recipe.setup_class and recipe.access_class and recipe.setup_class != recipe.access_class:
        reasons.append("setup/access class names differ")

    if reasons:
        if "missing-ir" in recipe.name or "hard-coded-order" in recipe.name or manual_fields:
            return "manual-review", tuple(reasons)
        return "review-needed", tuple(reasons)

    return "patch-sketch-candidate", ("matched descriptor source, setup/access context, and exact getter mappings",)


def format_classification(recipe: Recipe) -> list[str]:
    status, reasons = recipe_classification(recipe)
    lines = [f"      status: {status}"]
    for reason in reasons:
        lines.append(f"      reason: {reason}")
    return lines


def patch_sketch(recipe: Recipe) -> list[str]:
    classification, _ = recipe_classification(recipe)
    if classification != "patch-sketch-candidate":
        return []

    lines: list[str] = []
    if recipe.name.startswith("basis-index"):
        legacy_member = recipe.legacy_member_text or f"{recipe.legacy_index};"
        declaration_target = f" in {recipe.declaration_path}" if recipe.declaration_path else ""
        lines.extend([
            "Declaration sketch:",
            f"  - declaration target{declaration_target}",
            "  - include guidance:",
        ])
        lines.extend(format_include_changes(recipe))
        lines.extend([
            "  - remove legacy basis index member if it is no longer used:",
            f"      {legacy_member}",
            "  - add descriptor members:",
            "      panzer::BasisDescriptor basis_descriptor;",
            "      panzer::IntegrationDescriptor integration_descriptor;",
            "Constructor/setup sketch:",
            f"  - constructor context: {recipe.constructor_class or '<unknown>'}::{recipe.constructor_function or '<unknown>'}",
            f"  - Basis source: {recipe.basis_source_line or '?'}: {recipe.basis_source_text or '<not found>'}",
            f"  - IR source: {recipe.ir_source_line or '?'}: {recipe.ir_source_text or '<not found>'}",
            "  - initialize descriptors near the constructor source lines:",
        ])
        lines.extend(format_descriptor_initializers(recipe))
        lines.extend([
            "  - remove the getBasisIndex(...) assignment from postRegistrationSetup()",
            "evaluateFields sketch:",
        ])
    elif recipe.name.startswith("integration-index"):
        legacy_member = recipe.legacy_member_text or f"{recipe.legacy_index};"
        declaration_target = f" in {recipe.declaration_path}" if recipe.declaration_path else ""
        lines.extend([
            "Declaration sketch:",
            f"  - declaration target{declaration_target}",
            "  - include guidance:",
        ])
        lines.extend(format_include_changes(recipe))
        lines.extend([
            "  - remove legacy integration index member if it is no longer used:",
            f"      {legacy_member}",
            "  - add descriptor member:",
            "      panzer::IntegrationDescriptor integration_descriptor;",
            "Constructor/setup sketch:",
            f"  - constructor context: {recipe.constructor_class or '<unknown>'}::{recipe.constructor_function or '<unknown>'}",
            f"  - IR source: {recipe.ir_source_line or '?'}: {recipe.ir_source_text or '<not found>'}",
            "  - initialize descriptors near the constructor source lines:",
        ])
        lines.extend(format_descriptor_initializers(recipe))
        lines.extend([
            "  - remove the getIntegrationRuleIndex(...) assignment from postRegistrationSetup()",
            "evaluateFields sketch:",
        ])
    else:
        return []

    for line in format_suggested_handles(recipe):
        lines.append(f"  {line.strip()}")

    if recipe.legacy_alias:
        lines.extend([
            "Replacement sketch:",
            f"  - replace direct field uses through legacy alias `{recipe.legacy_alias}` with the suggested local handles",
        ])
    else:
        lines.extend([
            "Replacement sketch:",
            "  - replace direct `workset.bases[...]` / `workset.int_rules[...]` field aliases with the suggested local handles",
        ])
    lines.extend(format_replacements(recipe))

    return lines


def format_patch_sketch(recipe: Recipe) -> list[str]:
    return [f"      {line}" for line in patch_sketch(recipe)]


def format_legacy_pattern(recipe: Recipe) -> list[str]:
    lines: list[str] = []
    if recipe.setup_class or recipe.access_class:
        class_text = recipe.access_class or recipe.setup_class
        if recipe.setup_class and recipe.access_class and recipe.setup_class != recipe.access_class:
            class_text = f"setup={recipe.setup_class}, access={recipe.access_class}"
        lines.append(f"      evaluator/class: {class_text}")
    if recipe.setup_function:
        lines.append(f"      setup function: {recipe.setup_function}")
    if recipe.access_function:
        lines.append(f"      access function: {recipe.access_function}")
    if recipe.legacy_index:
        lines.append(f"      index: {recipe.legacy_index}")
    if recipe.legacy_alias:
        lines.append(f"      values alias: {recipe.legacy_alias}")
    if recipe.declaration_path:
        lines.append(f"      declaration file: {recipe.declaration_path}")
    if recipe.legacy_member_line:
        lines.append(f"      legacy member line: {recipe.legacy_member_line}: {recipe.legacy_member_text}")
    if recipe.constructor_function or recipe.constructor_class:
        constructor_text = recipe.constructor_function or "<unknown>"
        if recipe.constructor_class:
            constructor_text = f"{recipe.constructor_class}::{constructor_text}"
        lines.append(f"      constructor context: {constructor_text}")
    if recipe.basis_source_line:
        lines.append(f"      basis source line: {recipe.basis_source_line}: {recipe.basis_source_text}")
    if recipe.ir_source_line:
        lines.append(f"      IR source line: {recipe.ir_source_line}: {recipe.ir_source_text}")
    return lines


def format_suggested_handles(recipe: Recipe) -> list[str]:
    lines: list[str] = []
    owner = suggested_owner_handle(recipe)
    if owner:
        lines.append(f"      {owner}")
    for suggestion in suggested_field_handles(recipe):
        declaration = suggestion["declaration"]
        if declaration is None:
            lines.append(
                f"      // {suggestion['field']}: {suggestion['getter']}"
            )
        else:
            lines.append(f"      {declaration}")
    return lines


def recipe_to_json(recipe: Recipe) -> dict[str, object]:
    classification, classification_reasons = recipe_classification(recipe)
    return {
        "path": str(recipe.path),
        "line": recipe.line,
        "risk": recipe.risk,
        "recipe": recipe.name,
        "classification": classification,
        "classification_reasons": list(classification_reasons),
        "summary": recipe.summary,
        "hint": recipe.hint,
        "legacy_index": recipe.legacy_index,
        "legacy_alias": recipe.legacy_alias,
        "setup_function": recipe.setup_function,
        "access_function": recipe.access_function,
        "setup_class": recipe.setup_class,
        "access_class": recipe.access_class,
        "declaration_path": str(recipe.declaration_path) if recipe.declaration_path else None,
        "legacy_member_line": recipe.legacy_member_line,
        "legacy_member_text": recipe.legacy_member_text,
        "constructor_function": recipe.constructor_function,
        "constructor_class": recipe.constructor_class,
        "basis_source_line": recipe.basis_source_line,
        "basis_source_text": recipe.basis_source_text,
        "ir_source_line": recipe.ir_source_line,
        "ir_source_text": recipe.ir_source_text,
        "owner_handle": suggested_owner_handle(recipe),
        "fields": [
            {
                "name": field,
                "count": count,
                "getter": field_getter(field),
                "manual_review": field_getter(field).startswith("manual review") or "manual" in field_getter(field),
                "note": INTEGRATION_FIELD_NOTES.get(field),
            }
            for field, count in recipe.field_counts
        ],
        "phalanx_registration_fields": [
            {
                "name": field,
                "count": count,
                "getter": field_getter(field, phalanx_registration=True),
            }
            for field, count in recipe.phalanx_registration_field_counts
        ],
        "suggested_handles": suggested_field_handles(recipe),
        "suggested_replacements": suggested_replacements(recipe),
        "suggested_descriptor_initializers": suggested_descriptor_initializers(recipe),
        "suggested_include_changes": suggested_include_changes(recipe),
        "suggested_include_insertion_point": include_insertion_point(recipe.declaration_path),
        "patch_sketch": patch_sketch(recipe),
    }


def finding_to_json(finding: Finding) -> dict[str, object]:
    return {
        "path": str(finding.path),
        "line": finding.line,
        "risk": finding.rule.risk,
        "category": finding.rule.category,
        "rule": finding.rule.name,
        "hint": finding.rule.hint,
        "text": finding.text,
    }


def cache_hazard_to_json(hazard: CacheHazard) -> dict[str, object]:
    return {
        "path": str(hazard.path),
        "line": hazard.line,
        "risk": hazard.risk,
        "kind": hazard.kind,
        "summary": hazard.summary,
        "function": hazard.function,
        "class_name": hazard.class_name,
        "release_line": hazard.release_line,
        "handle_line": hazard.handle_line,
        "handle_name": hazard.handle_name,
        "text": hazard.text,
    }


def format_cache_hazard(hazard: CacheHazard) -> list[str]:
    context = hazard.function or "<unknown>"
    if hazard.class_name:
        context = f"{hazard.class_name}::{context}"
    lines = [
        f"  {hazard.path}:{hazard.line}: [{hazard.risk}/{hazard.kind}] {hazard.summary}",
        f"      function: {context}",
    ]
    if hazard.release_line is not None:
        lines.append(f"      releaseCachedData line: {hazard.release_line}")
    if hazard.handle_line is not None:
        lines.append(f"      retained handle: {hazard.handle_name} from line {hazard.handle_line}")
    if hazard.text:
        lines.append(f"      evidence: {hazard.text}")
    lines.append("      guidance: reacquire BasisValues2/IntegrationValues2 fields or views after releaseCachedData(); do not reuse handles captured before release.")
    return lines


def scan_recipes(path: pathlib.Path) -> list[Recipe]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"warning: could not read {path}: {exc}", file=sys.stderr)
        return []

    lines = text.splitlines()
    regions = find_function_regions(lines)
    recipes: list[Recipe] = []

    has_basis_parameter = '"Basis"' in text
    has_ir_parameter = '"IR"' in text
    has_integration_rule_parameter = bool(re.search(r"\bIntegrationRule\b", text))
    declaration_path = candidate_declaration_path(path)

    for assignment in find_assignments(lines, "getBasisIndex"):
        basis_alias_line, basis_alias = find_values_alias(lines, "bases", assignment.index)
        basis_access_line = first_index_access_line(lines, "bases", assignment.index)
        if basis_access_line is None:
            continue

        basis_setup_region = region_containing(regions, assignment.line)
        basis_access_region = region_containing(regions, basis_alias_line or basis_access_line)
        basis_search_text = region_text(lines, basis_access_region)
        basis_field_counts = field_counts_for_pattern(
            basis_search_text, BASIS_FIELD_GETTERS, "bases", assignment.index, basis_alias
        )
        basis_phalanx_registration_field_counts = count_phalanx_registration_fields(
            basis_search_text, BASIS_FIELD_GETTERS, "bases", assignment.index, basis_alias
        )
        if not basis_field_counts:
            continue

        legacy_member_line, legacy_member_text = find_legacy_member(declaration_path, assignment.index)
        constructor_context = find_constructor_context(
            lines, regions,
            (basis_access_region.class_name if basis_access_region else None)
            or (basis_setup_region.class_name if basis_setup_region else None)
        )

        if has_basis_parameter and has_ir_parameter:
            recipes.append(Recipe(
                path=path,
                line=min(assignment.line, basis_access_line),
                name="basis-index-to-descriptor-getter",
                risk="high",
                summary="Legacy getBasisIndex() plus workset.bases[...] and direct BasisValues2 field access.",
                hint=(
                    "Store BasisDescriptor and IntegrationDescriptor from constructor Basis/IR parameters; "
                    "remove the stored basis index; reacquire with getBasisValues(..., true); replace "
                    "basis fields with getBasisValues/getGradBasisValues/getBasisCoordinates using explicit "
                    "WeightBasis, CacheValues::Yes, and ForceRecompute::No."
                ),
                field_counts=basis_field_counts,
                phalanx_registration_field_counts=basis_phalanx_registration_field_counts,
                legacy_index=assignment.index,
                legacy_alias=basis_alias,
                setup_function=basis_setup_region.name if basis_setup_region else None,
                access_function=basis_access_region.name if basis_access_region else None,
                setup_class=basis_setup_region.class_name if basis_setup_region else None,
                access_class=basis_access_region.class_name if basis_access_region else None,
                declaration_path=declaration_path,
                legacy_member_line=legacy_member_line,
                legacy_member_text=legacy_member_text,
                constructor_function=constructor_context["constructor_function"],
                constructor_class=constructor_context["constructor_class"],
                basis_source_line=constructor_context["basis_source_line"],
                basis_source_text=constructor_context["basis_source_text"],
                ir_source_line=constructor_context["ir_source_line"],
                ir_source_text=constructor_context["ir_source_text"],
            ))
        else:
            recipes.append(Recipe(
                path=path,
                line=min(assignment.line, basis_access_line),
                name="basis-index-missing-ir-review",
                risk="high",
                summary="Legacy getBasisIndex() plus workset.bases[...] but no obvious Basis/IR constructor pair in this file.",
                hint=(
                    "Manual review needed to find the correct IntegrationDescriptor source. The basis descriptor may "
                    "come from an existing BasisIRLayout/PureBasis, but the integration descriptor may need to come "
                    "from a factory, surrounding parameter list, or a new evaluator constructor argument."
                ),
                field_counts=basis_field_counts,
                phalanx_registration_field_counts=basis_phalanx_registration_field_counts,
                legacy_index=assignment.index,
                legacy_alias=basis_alias,
                setup_function=basis_setup_region.name if basis_setup_region else None,
                access_function=basis_access_region.name if basis_access_region else None,
                setup_class=basis_setup_region.class_name if basis_setup_region else None,
                access_class=basis_access_region.class_name if basis_access_region else None,
                declaration_path=declaration_path,
                legacy_member_line=legacy_member_line,
                legacy_member_text=legacy_member_text,
                constructor_function=constructor_context["constructor_function"],
                constructor_class=constructor_context["constructor_class"],
                basis_source_line=constructor_context["basis_source_line"],
                basis_source_text=constructor_context["basis_source_text"],
                ir_source_line=constructor_context["ir_source_line"],
                ir_source_text=constructor_context["ir_source_text"],
            ))

    for assignment in find_assignments(lines, "getIntegrationRuleIndex"):
        integration_alias_line, integration_alias = find_values_alias(lines, "int_rules", assignment.index)
        integration_access_line = first_index_access_line(lines, "int_rules", assignment.index)
        if integration_access_line is None:
            continue

        integration_setup_region = region_containing(regions, assignment.line)
        integration_access_region = region_containing(regions, integration_alias_line or integration_access_line)
        integration_search_text = region_text(lines, integration_access_region)
        integration_field_counts = field_counts_for_pattern(
            integration_search_text, INTEGRATION_FIELD_GETTERS, "int_rules", assignment.index, integration_alias
        )
        if not integration_field_counts:
            continue

        legacy_member_line, legacy_member_text = find_legacy_member(declaration_path, assignment.index)
        constructor_context = find_constructor_context(
            lines, regions,
            (integration_access_region.class_name if integration_access_region else None)
            or (integration_setup_region.class_name if integration_setup_region else None)
        )

        if assignment.hard_coded_order:
            recipes.append(Recipe(
                path=path,
                line=min(assignment.line, integration_access_line),
                name="integration-index-hard-coded-order-review",
                risk="high",
                summary="Legacy getIntegrationRuleIndex() uses a hard-coded order with workset.int_rules[...] and direct IntegrationValues2 field access.",
                hint=(
                    "Manual review needed before replacing the index with an IntegrationDescriptor. If a constructor "
                    "IntegrationRule is available, confirm that the hard-coded order describes the same rule before "
                    "storing the descriptor and using getIntegrationValues(..., true)."
                ),
                field_counts=integration_field_counts,
                legacy_index=assignment.index,
                legacy_alias=integration_alias,
                setup_function=integration_setup_region.name if integration_setup_region else None,
                access_function=integration_access_region.name if integration_access_region else None,
                setup_class=integration_setup_region.class_name if integration_setup_region else None,
                access_class=integration_access_region.class_name if integration_access_region else None,
                declaration_path=declaration_path,
                legacy_member_line=legacy_member_line,
                legacy_member_text=legacy_member_text,
                constructor_function=constructor_context["constructor_function"],
                constructor_class=constructor_context["constructor_class"],
                basis_source_line=constructor_context["basis_source_line"],
                basis_source_text=constructor_context["basis_source_text"],
                ir_source_line=constructor_context["ir_source_line"],
                ir_source_text=constructor_context["ir_source_text"],
            ))
        elif has_ir_parameter or has_integration_rule_parameter:
            recipes.append(Recipe(
                path=path,
                line=min(assignment.line, integration_access_line),
                name="integration-index-to-descriptor-getter",
                risk="high",
                summary="Legacy getIntegrationRuleIndex() plus workset.int_rules[...] and direct IntegrationValues2 field access.",
                hint=(
                    "Store an IntegrationDescriptor from the constructor IntegrationRule/IR parameter; remove the stored "
                    "integration index; reacquire with getIntegrationValues(..., true); replace direct fields with "
                    "IntegrationValues2 getters where available. Emit manual review when no exact getter mapping exists."
                ),
                field_counts=integration_field_counts,
                legacy_index=assignment.index,
                legacy_alias=integration_alias,
                setup_function=integration_setup_region.name if integration_setup_region else None,
                access_function=integration_access_region.name if integration_access_region else None,
                setup_class=integration_setup_region.class_name if integration_setup_region else None,
                access_class=integration_access_region.class_name if integration_access_region else None,
                declaration_path=declaration_path,
                legacy_member_line=legacy_member_line,
                legacy_member_text=legacy_member_text,
                constructor_function=constructor_context["constructor_function"],
                constructor_class=constructor_context["constructor_class"],
                basis_source_line=constructor_context["basis_source_line"],
                basis_source_text=constructor_context["basis_source_text"],
                ir_source_line=constructor_context["ir_source_line"],
                ir_source_text=constructor_context["ir_source_text"],
            ))
        else:
            recipes.append(Recipe(
                path=path,
                line=min(assignment.line, integration_access_line),
                name="integration-index-missing-ir-review",
                risk="high",
                summary="Legacy getIntegrationRuleIndex() plus workset.int_rules[...] but no obvious IR parameter in this file.",
                hint=(
                    "Manual review needed to find the correct IntegrationDescriptor source. Some files only store an "
                    "integration order or cubature degree; those may need a propagated IntegrationRule/IntegrationDescriptor "
                    "or a local descriptor constructed from order/type/side information."
                ),
                field_counts=integration_field_counts,
                legacy_index=assignment.index,
                legacy_alias=integration_alias,
                setup_function=integration_setup_region.name if integration_setup_region else None,
                access_function=integration_access_region.name if integration_access_region else None,
                setup_class=integration_setup_region.class_name if integration_setup_region else None,
                access_class=integration_access_region.class_name if integration_access_region else None,
                declaration_path=declaration_path,
                legacy_member_line=legacy_member_line,
                legacy_member_text=legacy_member_text,
                constructor_function=constructor_context["constructor_function"],
                constructor_class=constructor_context["constructor_class"],
                basis_source_line=constructor_context["basis_source_line"],
                basis_source_text=constructor_context["basis_source_text"],
                ir_source_line=constructor_context["ir_source_line"],
                ir_source_text=constructor_context["ir_source_text"],
            ))

    return recipes


def run_self_test() -> int:
    sample = """
void ExampleEvaluator::evaluateFields(typename Traits::EvalData workset)
{
  auto& bv = *workset.bases[basis_index];
  auto weighted_basis = bv.weighted_basis_scalar;
  auto old_view = weighted_basis.get_view();
  bv.releaseCachedData();
  weighted_basis(0,0,0) = 0.0;
  bv.grad_basis(0,0,0,0) = 0.0;
  auto new_view = bv.weighted_grad_basis.get_view();
}
""".strip().splitlines()
    hazards = scan_cache_hazards_from_lines(pathlib.Path("<self-test-cache-hazard.cpp>"), sample)
    kinds = Counter(hazard.kind for hazard in hazards)
    expected = {
        "retained-handle-after-release",
        "direct-field-after-release",
        "view-after-release",
    }
    missing = expected - set(kinds)
    if missing:
        print("self-test failed: missing cache hazard kinds: " + ", ".join(sorted(missing)), file=sys.stderr)
        for hazard in hazards:
            for line in format_cache_hazard(hazard):
                print(line, file=sys.stderr)
        return 1

    registration_sample = """
    this->addEvaluatedField(basisValues->basis_scalar);
    this->addNonConstDependentField(basisValues->basis_vector);
    this->utils.setFieldData(basisValues->grad_basis,fm);
    auto values = basisValues->weighted_basis_scalar;
    """
    registration_counts = count_phalanx_registration_fields(
        registration_sample, BASIS_FIELD_GETTERS, "bases", None, "basisValues"
    )
    registration_count_map = count_dict(registration_counts)
    expected_registration_counts = {
        "basis_scalar": 1,
        "basis_vector": 1,
        "grad_basis": 1,
    }
    if registration_count_map != expected_registration_counts:
        print(
            "self-test failed: Phalanx registration field context detection mismatch: "
            + repr(registration_count_map),
            file=sys.stderr,
        )
        return 1

    recipe = Recipe(
        path=pathlib.Path("<self-test-phalanx-registration.cpp>"),
        line=1,
        name="basis-index-to-descriptor-getter",
        risk="high",
        summary="self-test",
        hint="self-test",
        field_counts=(("basis_scalar", 1), ("weighted_basis_scalar", 1)),
        phalanx_registration_field_counts=(("basis_scalar", 1),),
    )
    handles = {str(item["field"]): item for item in suggested_field_handles(recipe)}
    if handles["basis_scalar"]["getter"] != "getBasisValuesForPhalanxEvaluatorRegistration()":
        print("self-test failed: basis_scalar did not use the Phalanx registration getter", file=sys.stderr)
        return 1
    if handles["weighted_basis_scalar"]["getter"] != "getBasisValues(panzer::WeightBasis::Yes)":
        print("self-test failed: weighted_basis_scalar did not use the lazy value getter", file=sys.stderr)
        return 1

    const_fallout_sample = [
        "void foo(const PHX::MDField<double,panzer::Cell,panzer::IP,panzer::Dim> & points);",
        "void bar(const PHX::MDField<const double,panzer::Cell,panzer::IP,panzer::Dim> & points);",
    ]
    const_fallout_findings = []
    for line_no, line in enumerate(const_fallout_sample, start=1):
        for rule in RULES:
            if rule.name == "integration-getter-nonconst-mdfield-ref" and rule.pattern.search(line):
                const_fallout_findings.append(line_no)
    if const_fallout_findings != [1]:
        print(
            "self-test failed: const getter fallout detection expected only line 1, got "
            + repr(const_fallout_findings),
            file=sys.stderr,
        )
        return 1

    point_mapping_expectations = {
        "cub_points": "getUniformCubaturePointsRef()",
        "side_cub_points": "manual review: getUniformSideCubaturePointsRef() is shape-preserving but deprecated; getCubaturePointsRef() changes rank/semantics",
        "cub_weights": "getUniformCubatureWeightsRef()",
        "ip_coordinates": "getCubaturePoints()",
        "ref_ip_coordinates": "getCubaturePointsRef()",
    }
    for field, getter in point_mapping_expectations.items():
        if field_getter(field) != getter:
            print(
                f"self-test failed: integration point mapping for {field} expected {getter}, got {field_getter(field)}",
                file=sys.stderr,
            )
            return 1

    private_vs_descriptor_sample = [
        "auto jac = workset.int_rules[ir_index]->jac;",
        "auto points = workset.int_rules[ir_index]->getCubaturePoints();",
        "auto basis = workset.bases[basis_index]->basis_scalar;",
        "auto basis_values = workset.bases[basis_index]->getBasisValues();",
    ]
    private_descriptor_hits: dict[str, list[int]] = defaultdict(list)
    for line_no, line in enumerate(private_vs_descriptor_sample, start=1):
        for rule in RULES:
            if rule.name in {
                "workset-int-rules-private-field",
                "workset-bases-private-field",
                "workset-int-rules-index",
                "workset-bases-index",
            } and rule.pattern.search(line):
                private_descriptor_hits[rule.name].append(line_no)
    expected_private_descriptor_hits = {
        "workset-int-rules-private-field": [1],
        "workset-bases-private-field": [3],
        "workset-int-rules-index": [1, 2],
        "workset-bases-index": [3, 4],
    }
    if dict(private_descriptor_hits) != expected_private_descriptor_hits:
        print(
            "self-test failed: private-field breakage vs descriptor-indexing split mismatch: "
            + repr(dict(private_descriptor_hits)),
            file=sys.stderr,
        )
        return 1

    temporary_getter_sample = [
        "auto bad = Kokkos::subview(iv.getJacobian().get_view(), 0, 0, Kokkos::ALL(), Kokkos::ALL());",
        "auto jacobian = iv.getJacobian();",
        "auto good = Kokkos::subview(jacobian.get_view(), 0, 0, Kokkos::ALL(), Kokkos::ALL());",
    ]
    temporary_getter_findings = []
    for line_no, line in enumerate(temporary_getter_sample, start=1):
        for rule in RULES:
            if rule.name == "temporary-getter-view-hazard" and rule.pattern.search(line):
                temporary_getter_findings.append(line_no)
    if temporary_getter_findings != [1]:
        print(
            "self-test failed: temporary getter view hazard expected only line 1, got "
            + repr(temporary_getter_findings),
            file=sys.stderr,
        )
        return 1

    descriptor_recipe = Recipe(
        path=pathlib.Path("<self-test-descriptor.cpp>"),
        line=1,
        name="integration-index-to-descriptor-getter",
        risk="high",
        summary="self-test",
        hint="self-test",
        ir_source_text="const panzer::IntegrationRule & ir,",
    )
    descriptor_initializers = suggested_descriptor_initializers(descriptor_recipe)
    if not descriptor_initializers or descriptor_initializers[0]["expression"] != "integration_descriptor = panzer::IntegrationDescriptor(ir.getOrder(), ir.getType(), ir.getSide());":
        print(
            "self-test failed: IntegrationRule reference descriptor initializer mismatch: "
            + repr(descriptor_initializers),
            file=sys.stderr,
        )
        return 1

    with tempfile.TemporaryDirectory() as tmpdir:
        direct_path = pathlib.Path(tmpdir) / "direct.cpp"
        direct_path.write_text(
            "void f(auto workset) {\n"
            "  auto w = workset.int_rules[ir_index]->weighted_measure;\n"
            "  auto g = workset.int_rules[ir_index]->norm_contravarient(cell,point);\n"
            "  auto b = workset.bases[basis_index]->weighted_grad_basis.get_view();\n"
            "}\n",
            encoding="utf-8",
        )
        if not _replace_direct_indexed_field_getters(direct_path):
            print("self-test failed: direct indexed field replacement reported no change", file=sys.stderr)
            return 1
        direct_text = direct_path.read_text(encoding="utf-8")
        expected_direct_fragments = [
            "workset.int_rules[ir_index]->getWeightedMeasure()",
            "workset.int_rules[ir_index]->getNormContravarientMatrix(true)(cell,point)",
            "workset.bases[basis_index]->getGradBasisValues(panzer::WeightBasis::Yes).get_view()",
        ]
        missing_direct = [fragment for fragment in expected_direct_fragments if fragment not in direct_text]
        if missing_direct:
            print(
                "self-test failed: direct indexed field replacement missing fragments: "
                + repr(missing_direct)
                + " in "
                + direct_text,
                file=sys.stderr,
            )
            return 1

    print("Self-test passed: cache lifecycle hazard sample produced expected advisory kinds.")
    for kind, count in kinds.most_common():
        print(f"  {kind}: {count}")
    print("Self-test passed: Phalanx registration field contexts use registration getters.")
    print("Self-test passed: non-const IntegrationValues2 MDField references are diagnosed.")
    print("Self-test passed: integration point field mappings are precise.")
    print("Self-test passed: private-field breakage and descriptor-indexing advisories are separated.")
    print("Self-test passed: temporary getter view/subview hazards are diagnosed.")
    print("Self-test passed: IntegrationRule references initialize descriptors with object access.")
    print("Self-test passed: direct indexed private-field patch sketches use explicit getters.")
    return 0


def _line_ending(lines: list[str]) -> str:
    for line in lines:
        if line.endswith("\r\n"):
            return "\r\n"
    return "\n"


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _index_alternatives(index: str | None) -> list[str]:
    if not index:
        return []
    alts = [index]
    if index.startswith("this->"):
        alts.append(index.removeprefix("this->"))
    else:
        alts.append("this->" + index)
    return list(dict.fromkeys(alts))


def _container_pattern(recipe: Recipe) -> str:
    if recipe.name.startswith("basis-index"):
        return r"(?:this->wda\(\s*workset\s*\)|wda\(\s*workset\s*\)|workset)\.bases"
    return r"(?:this->wda\(\s*workset\s*\)|wda\(\s*workset\s*\)|workset)\.int_rules"


def _indexed_field_pattern(recipe: Recipe, field: str) -> re.Pattern[str]:
    index_patterns = [re.escape(alt).replace(r"\ ", r"\s*") for alt in _index_alternatives(recipe.legacy_index)]
    index_pattern = "|".join(index_patterns) if index_patterns else r"[^\]]+"
    return re.compile(
        _container_pattern(recipe)
        + r"\s*\[\s*(?:" + index_pattern + r")\s*\]\s*->\s*"
        + re.escape(field)
        + r"\b"
    )


def _alias_field_pattern(alias: str, field: str) -> re.Pattern[str]:
    return re.compile(r"\b" + re.escape(alias) + r"\s*\.\s*" + re.escape(field) + r"\b")


def _is_alias_definition(line: str, recipe: Recipe) -> bool:
    if not recipe.legacy_alias:
        return False
    if not re.search(r"\b" + re.escape(recipe.legacy_alias) + r"\b\s*=", line):
        return False
    return bool(re.search(_container_pattern(recipe) + r"\s*\[", line))


def _is_index_assignment(line: str, recipe: Recipe) -> bool:
    if not recipe.legacy_index:
        return False
    member = recipe.legacy_index.removeprefix("this->")
    if member not in line or "panzer::get" not in line:
        return False
    if recipe.name.startswith("basis-index"):
        return "getBasisIndex" in line
    return "getIntegrationRuleIndex" in line


def _find_region(lines: list[str], function: str | None, class_name: str | None) -> FunctionRegion | None:
    if function is None:
        return None
    for region in find_function_regions([line.rstrip("\n\r") for line in lines]):
        if region.name != function:
            continue
        if class_name is not None and region.class_name != class_name:
            continue
        return region
    for region in find_function_regions([line.rstrip("\n\r") for line in lines]):
        if region.name == function:
            return region
    return None


def _insert_includes(path: pathlib.Path, recipe: Recipe) -> bool:
    if recipe.declaration_path is None or recipe.declaration_path != path:
        return False
    changes = suggested_include_changes(recipe)
    if not changes:
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    ending = _line_ending(lines)
    point = include_insertion_point(path)
    if point is None:
        return False
    insert_at = int(point["line"])
    additions = []
    for change in changes:
        header = str(change["include"])
        if header not in text:
            additions.append(header + ending)
    if not additions:
        return False
    lines[insert_at:insert_at] = additions
    path.write_text("".join(lines), encoding="utf-8")
    return True


def _replace_legacy_member(path: pathlib.Path, recipe: Recipe) -> bool:
    if recipe.declaration_path is None or recipe.declaration_path != path or not recipe.legacy_member_text:
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    ending = _line_ending(lines)
    member_name = (recipe.legacy_index or "").removeprefix("this->")
    changed = False
    for i, line in enumerate(lines):
        if member_name not in line:
            continue
        if recipe.legacy_member_text.strip() not in line.strip():
            continue
        indent = _indent_of(line)
        if recipe.name.startswith("basis-index"):
            replacement = [
                f"{indent}panzer::BasisDescriptor basis_descriptor;{ending}",
                f"{indent}panzer::IntegrationDescriptor integration_descriptor;{ending}",
            ]
        else:
            replacement = [f"{indent}panzer::IntegrationDescriptor integration_descriptor;{ending}"]
        lines[i:i + 1] = replacement
        changed = True
        break
    if changed:
        path.write_text("".join(lines), encoding="utf-8")
    return changed


CONST_FIELD_MEMBER_TYPES = {
    "basis_scalar": "PHX::MDField<const double, Cell, BASIS, IP>",
    "weighted_basis_scalar": "PHX::MDField<const double, Cell, BASIS, IP>",
    "grad_basis": "PHX::MDField<const double, Cell, BASIS, IP, Dim>",
    "weighted_grad_basis": "PHX::MDField<const double, Cell, BASIS, IP, Dim>",
    "basis_coordinates_ref": "PHX::MDField<const double, Cell, BASIS, Dim>",
    "basis_coordinates": "PHX::MDField<const double, Cell, BASIS, Dim>",
}


def _member_field_aliases(recipe: Recipe) -> dict[str, str]:
    text = recipe.path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    region = _find_region(lines, recipe.access_function, recipe.access_class)
    aliases: dict[str, str] = {}
    search_start = region.start - 1 if region else 0
    search_end = region.end if region else len(lines)
    registration_counts = count_dict(recipe.phalanx_registration_field_counts)
    for field, count in recipe.field_counts:
        if registration_counts.get(field, 0) == count:
            continue
        for line in lines[search_start:search_end]:
            name = _field_alias_declaration_name(line, recipe, field)
            if name is None:
                continue
            lhs = line.split("=", 1)[0]
            match = re.search(r"\bthis\s*->\s*([A-Za-z_][A-Za-z0-9_]*)\s*$", lhs.strip())
            if match:
                aliases[match.group(1)] = field
            break
    return aliases


def _replace_field_member_declarations(path: pathlib.Path, recipe: Recipe) -> bool:
    if recipe.declaration_path is None or recipe.declaration_path != path:
        return False
    member_fields = _member_field_aliases(recipe)
    if not member_fields:
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    changed = False
    for i, line in enumerate(lines):
        for member_name, field in member_fields.items():
            const_type = CONST_FIELD_MEMBER_TYPES.get(field)
            if const_type is None or not re.search(r"\b" + re.escape(member_name) + r"\s*;", line):
                continue
            indent = _indent_of(line)
            trailing = ""
            if "//!<" in line:
                trailing = " " + line.split("//!<", 1)[1].rstrip("\r\n")
                trailing = " //!<" + trailing.lstrip()
            ending = "\r\n" if line.endswith("\r\n") else "\n"
            replacement = f"{indent}{const_type} {member_name};"
            if trailing:
                replacement = replacement.ljust(70) + trailing
            lines[i] = replacement + ending
            changed = True
            break
    if changed:
        path.write_text("".join(lines), encoding="utf-8")
    return changed


def _insert_descriptor_initializers(lines: list[str], recipe: Recipe) -> bool:
    initializers = [str(s["expression"]) for s in suggested_descriptor_initializers(recipe) if not s["manual_review"]]
    if not initializers:
        return False
    current_text = "".join(lines)
    initializers = [expr for expr in initializers if expr not in current_text]
    if not initializers:
        return False
    insert_after = max(line for line in (recipe.basis_source_line, recipe.ir_source_line) if line is not None)
    if insert_after < 1 or insert_after > len(lines):
        return False
    ending = _line_ending(lines)
    indent = _indent_of(lines[insert_after - 1])
    additions = [f"{indent}{expr}{ending}" for expr in initializers]
    lines[insert_after:insert_after] = additions
    return True


def _remove_index_assignment(lines: list[str], recipe: Recipe) -> bool:
    changed = False
    region = _find_region(lines, recipe.setup_function, recipe.setup_class)
    start = region.start - 1 if region else 0
    end = region.end if region else len(lines)
    kept = []
    for offset, line in enumerate(lines[start:end], start=start):
        if _is_index_assignment(line, recipe):
            changed = True
            continue
        kept.append(line)
    if changed:
        lines[start:end] = kept
    return changed


def _legacy_access_line(line: str, recipe: Recipe) -> bool:
    if recipe.legacy_alias and any(_alias_field_pattern(recipe.legacy_alias, field).search(line) for field, _ in recipe.field_counts):
        return True
    return any(_indexed_field_pattern(recipe, field).search(line) for field, _ in recipe.field_counts)


def _line_uses_legacy_field(line: str, recipe: Recipe, field: str) -> bool:
    if recipe.legacy_alias and _alias_field_pattern(recipe.legacy_alias, field).search(line):
        return True
    return bool(_indexed_field_pattern(recipe, field).search(line))


def _field_alias_declaration_name(line: str, recipe: Recipe, field: str) -> str | None:
    if not _line_uses_legacy_field(line, recipe, field):
        return None
    if "=" not in line or ";" not in line:
        return None
    if any(op in line for op in ("+=", "-=", "*=", "/=", "==", "!=", "<=", ">=")):
        return None
    lhs, rhs = line.split("=", 1)
    rhs = rhs.strip().rstrip(";").strip()
    # Only treat declarations that alias the whole field as field handles. Element
    # access like `x = bv.basis_scalar(cell,i,qp)` must keep the element access.
    if recipe.legacy_alias:
        alias_field = re.compile(r"^\*?\s*" + re.escape(recipe.legacy_alias) + r"\s*\.\s*" + re.escape(field) + r"\s*$")
        if not alias_field.match(rhs):
            return None
    else:
        index_patterns = [re.escape(alt).replace(r"\ ", r"\s*") for alt in _index_alternatives(recipe.legacy_index)]
        index_pattern = "|".join(index_patterns) if index_patterns else r"[^\]]+"
        indexed_field = re.compile(
            r"^\*?\s*" + _container_pattern(recipe) + r"\s*\[\s*(?:" + index_pattern + r")\s*\]\s*->\s*" + re.escape(field) + r"\s*$"
        )
        if not indexed_field.match(rhs):
            return None
    match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*$", lhs)
    return match.group(1) if match else None


def _field_handle_names_in_region(lines: list[str], recipe: Recipe, region: FunctionRegion | None) -> dict[str, str]:
    names: dict[str, str] = {}
    if region is None:
        return names
    for field, _ in recipe.field_counts:
        for line in lines[region.start - 1:region.end]:
            name = _field_alias_declaration_name(line, recipe, field)
            if name is not None:
                names[field] = name
                break
    return names


def _insert_local_handles(lines: list[str], recipe: Recipe) -> bool:
    region = _find_region(lines, recipe.access_function, recipe.access_class)
    if region is None:
        return False
    region_text_now = "".join(lines[region.start - 1:region.end])
    owner = suggested_owner_handle(recipe)
    if owner is None or owner in region_text_now:
        return False
    insert_at = None
    for i in range(region.start - 1, region.end):
        if _is_alias_definition(lines[i], recipe) or _legacy_access_line(lines[i], recipe):
            insert_at = i
            break
    if insert_at is None:
        return False
    ending = _line_ending(lines)
    indent = _indent_of(lines[insert_at])
    local_names = _field_handle_names_in_region(lines, recipe, region)
    additions = [f"{indent}{owner}{ending}"]
    for suggestion in suggested_field_handles(recipe):
        field = str(suggestion["field"])
        if field in local_names:
            continue
        declaration = suggestion.get("declaration")
        if declaration:
            additions.append(f"{indent}{declaration}{ending}")
    lines[insert_at:insert_at] = additions
    return True


def _replace_legacy_accesses(lines: list[str], recipe: Recipe) -> bool:
    changed = False
    region = _find_region(lines, recipe.access_function, recipe.access_class)
    local_names = _field_handle_names_in_region(lines, recipe, region)
    owner_name = "basis_values" if recipe.name.startswith("basis-index") else "integration_values"

    # Remove simple local aliases to workset.bases[]/int_rules[] after inserting descriptor handles.
    for i, line in enumerate(list(lines)):
        if _is_alias_definition(line, recipe):
            lines[i] = ""
            changed = True

    # Replace typed field-reference declarations with auto declarations from the accessor.
    for suggestion in suggested_field_handles(recipe):
        field = str(suggestion["field"])
        if suggestion.get("manual_review"):
            continue
        getter = str(suggestion["getter"])
        local_name = local_names.get(field)
        if local_name is None:
            continue
        for i, line in enumerate(lines):
            if _field_alias_declaration_name(line, recipe, field) != local_name:
                continue
            indent = _indent_of(line)
            ending = "\r\n" if line.endswith("\r\n") else "\n"
            lhs = line.split("=", 1)[0].strip()
            if "this->" in lhs:
                lines[i] = f"{indent}{lhs} = {owner_name}.{getter};{ending}"
            else:
                qualifier = "auto" if suggestion.get("phalanx_registration") else "const auto"
                lines[i] = f"{indent}{qualifier} {local_name} = {owner_name}.{getter};{ending}"
            changed = True

    for suggestion in suggested_field_handles(recipe):
        field = str(suggestion["field"])
        if suggestion.get("manual_review"):
            continue
        handle = local_names.get(field, str(suggestion["handle"]))
        indexed = _indexed_field_pattern(recipe, field)
        for i, line in enumerate(lines):
            new_line = indexed.sub(handle, line)
            if recipe.legacy_alias:
                new_line = _alias_field_pattern(recipe.legacy_alias, field).sub(handle, new_line)
            if new_line != line:
                lines[i] = new_line
                changed = True

    # If the old alias itself is passed around after field replacement, pass the
    # descriptor-reacquired values object instead of leaving an undeclared alias.
    if recipe.legacy_alias:
        alias_pattern = re.compile(r"\b" + re.escape(recipe.legacy_alias) + r"\b")
        search_start = region.start - 1 if region else 0
        search_end = region.end if region else len(lines)
        for i in range(search_start, min(search_end, len(lines))):
            if _is_alias_definition(lines[i], recipe):
                continue
            new_line = alias_pattern.sub(owner_name, lines[i])
            if new_line != lines[i]:
                lines[i] = new_line
                changed = True
    return changed


def _replace_direct_indexed_field_getters(path: pathlib.Path) -> bool:
    """Apply local getter substitutions for direct workset indexed fields.

    This deliberately does not introduce descriptors or touch arbitrary aliases. It
    only rewrites the private-field portion of expressions such as
    `workset.int_rules[i]->weighted_measure` to the corresponding getter call so
    applications get out of the private-data breakage while still receiving the
    descriptor-index advisory for the remaining `workset.int_rules[i]` access.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    changed = False

    safe_integration_getters = {
        field: getter
        for field, getter in INTEGRATION_FIELD_GETTERS.items()
        if not getter.startswith("manual review")
    }
    safe_basis_getters = {
        field: getter
        for field, getter in BASIS_FIELD_GETTERS.items()
        if not getter.startswith("manual review")
    }

    integration_pattern = re.compile(
        r"(?P<prefix>\(?\s*\b(?:this->wda\(\s*workset\s*\)|wda\(\s*workset\s*\)|workset)\.int_rules\s*\[[^\]]+\]\s*\)?\s*->\s*)"
        r"(?P<field>" + "|".join(re.escape(field) for field in safe_integration_getters) + r")\b"
    )
    basis_pattern = re.compile(
        r"(?P<prefix>\(?\s*\b(?:this->wda\(\s*workset\s*\)|wda\(\s*workset\s*\)|workset)\.bases\s*\[[^\]]+\]\s*\)?\s*->\s*)"
        r"(?P<field>" + "|".join(re.escape(field) for field in safe_basis_getters) + r")\b"
    )

    def replace_integration(match: re.Match[str]) -> str:
        return match.group("prefix") + safe_integration_getters[match.group("field")]

    def replace_basis(match: re.Match[str]) -> str:
        return match.group("prefix") + safe_basis_getters[match.group("field")]

    for i, line in enumerate(lines):
        if line.lstrip().startswith("//"):
            continue
        new_line = integration_pattern.sub(replace_integration, line)
        new_line = basis_pattern.sub(replace_basis, new_line)
        if new_line != line:
            lines[i] = new_line
            changed = True

    if changed:
        path.write_text("".join(lines), encoding="utf-8")
    return changed


def apply_patch_sketches(recipes: list[Recipe], findings: list[Finding] | None = None) -> tuple[int, list[str]]:
    applied: list[str] = []
    findings = findings or []
    candidates = [r for r in recipes if recipe_classification(r)[0] == "patch-sketch-candidate"]

    declaration_recipes: dict[pathlib.Path, list[Recipe]] = defaultdict(list)
    implementation_recipes: dict[pathlib.Path, list[Recipe]] = defaultdict(list)
    for recipe in candidates:
        if recipe.declaration_path is not None:
            declaration_recipes[recipe.declaration_path].append(recipe)
        implementation_recipes[recipe.path].append(recipe)

    changed_paths: set[pathlib.Path] = set()
    for path, path_recipes in declaration_recipes.items():
        changed = False
        for recipe in path_recipes:
            changed = _insert_includes(path, recipe) or changed
            changed = _replace_legacy_member(path, recipe) or changed
            changed = _replace_field_member_declarations(path, recipe) or changed
        if changed:
            changed_paths.add(path)

    for path, path_recipes in implementation_recipes.items():
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        changed = False
        for recipe in path_recipes:
            changed = _insert_descriptor_initializers(lines, recipe) or changed
            changed = _remove_index_assignment(lines, recipe) or changed
            changed = _insert_local_handles(lines, recipe) or changed
            changed = _replace_legacy_accesses(lines, recipe) or changed
        if changed:
            path.write_text("".join(lines), encoding="utf-8")
            changed_paths.add(path)

    direct_field_paths = {
        finding.path
        for finding in findings
        if finding.rule.name in {"workset-int-rules-private-field", "workset-bases-private-field"}
    }
    for path in sorted(direct_field_paths):
        if _replace_direct_indexed_field_getters(path):
            changed_paths.add(path)

    for path in sorted(changed_paths):
        applied.append(str(path))
    return len(candidates), applied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", type=pathlib.Path, help="Source roots to scan")
    parser.add_argument("--self-test", action="store_true", help="Run built-in scanner self-tests and exit")
    parser.add_argument("--format", choices=("text", "tsv", "json"), default="text")
    parser.add_argument("--max-line-length", type=int, default=180)
    parser.add_argument("--recipes-only", action="store_true", help="Only print migration recipe candidates")
    parser.add_argument("--patch-sketches-only", action="store_true", help="Only print non-applied patch sketches for patch-sketch candidates")
    parser.add_argument("--apply-patch-sketches", action="store_true", help="Apply patch-sketch candidates in place")
    parser.add_argument("--cache-hazards-only", action="store_true", help="Only print releaseCachedData() lifecycle hazard candidates")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()
    if not args.roots:
        parser.error("at least one source root is required unless --self-test is used")

    all_findings: list[Finding] = []
    all_recipes: list[Recipe] = []
    all_cache_hazards: list[CacheHazard] = []
    for root in args.roots:
        if not root.exists():
            print(f"warning: root does not exist: {root}", file=sys.stderr)
            continue
        files = [root] if root.is_file() else list(iter_files(root))
        for path in files:
            all_findings.extend(scan_file(path))
            all_recipes.extend(scan_recipes(path))
            all_cache_hazards.extend(scan_cache_hazards(path))

    all_findings.sort(key=lambda f: (str(f.path), f.line, f.rule.name))
    all_recipes.sort(key=lambda r: (str(r.path), r.line, r.name))
    all_cache_hazards.sort(key=lambda h: (str(h.path), h.line, h.kind))
    if args.patch_sketches_only:
        args.recipes_only = True

    if args.apply_patch_sketches:
        candidate_count, changed_paths = apply_patch_sketches(all_recipes, all_findings)
        print(f"Applied patch-sketch candidates: {candidate_count}")
        print(f"Changed files: {len(changed_paths)}")
        for path in changed_paths:
            print(f"  {path}")
        return 0

    if args.format == "json":
        finding_counts = Counter(f.rule.name for f in all_findings)
        finding_risk_counts = Counter(f.rule.risk for f in all_findings)
        finding_category_counts = Counter(f.rule.category for f in all_findings)
        recipe_counts = Counter(r.name for r in all_recipes)
        cache_hazard_counts = Counter(h.kind for h in all_cache_hazards)
        cache_hazard_risk_counts = Counter(h.risk for h in all_cache_hazards)
        recipes_for_output = [
            recipe for recipe in all_recipes
            if not args.patch_sketches_only or recipe_classification(recipe)[0] == "patch-sketch-candidate"
        ]
        payload: dict[str, object] = {
            "summary": {
                "findings": len(all_findings),
                "recipes": len(all_recipes),
                "output_recipes": len(recipes_for_output),
                "finding_risk_counts": dict(sorted(finding_risk_counts.items())),
                "finding_category_counts": dict(sorted(finding_category_counts.items())),
                "finding_rule_counts": dict(finding_counts.most_common()),
                "recipe_counts": dict(recipe_counts.most_common()),
                "cache_hazards": len(all_cache_hazards),
                "cache_hazard_counts": dict(cache_hazard_counts.most_common()),
                "cache_hazard_risk_counts": dict(sorted(cache_hazard_risk_counts.items())),
            },
            "recipes": [] if args.cache_hazards_only else [recipe_to_json(recipe) for recipe in recipes_for_output],
        }
        if args.cache_hazards_only:
            payload["cache_hazards"] = [cache_hazard_to_json(hazard) for hazard in all_cache_hazards]
        elif not args.recipes_only:
            payload["findings"] = [finding_to_json(finding) for finding in all_findings]
            payload["cache_hazards"] = [cache_hazard_to_json(hazard) for hazard in all_cache_hazards]
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.format == "tsv":
        if args.cache_hazards_only:
            print("path\tline\trisk\tkind\tsummary\tclass_name\tfunction\trelease_line\thandle_line\thandle_name\ttext")
            for h in all_cache_hazards:
                print(
                    f"{h.path}\t{h.line}\t{h.risk}\t{h.kind}\t{h.summary}\t"
                    f"{h.class_name or ''}\t{h.function or ''}\t{h.release_line or ''}\t"
                    f"{h.handle_line or ''}\t{h.handle_name or ''}\t{h.text or ''}"
                )
        elif args.recipes_only:
            print("path\tline\trisk\trecipe\tclassification\tclassification_reasons\tsetup_class\taccess_class\tsetup_function\taccess_function\tlegacy_index\tlegacy_alias\tdeclaration_path\tlegacy_member_line\tlegacy_member_text\tconstructor_class\tconstructor_function\tbasis_source_line\tbasis_source_text\tir_source_line\tir_source_text\tfields\tsuggested_replacements\tsuggested_descriptor_initializers\tsuggested_include_changes\tsuggested_include_insertion_point\thint\tsummary")
            for r in all_recipes:
                if args.patch_sketches_only and recipe_classification(r)[0] != "patch-sketch-candidate":
                    continue
                fields = "; ".join(
                    f"{field}:{count}" for field, count in r.field_counts
                )
                replacements = "; ".join(
                    f"{replacement['legacy']}->{replacement['replacement'] or 'manual review'}"
                    for replacement in suggested_replacements(r)
                )
                initializers = "; ".join(
                    f"{initializer['descriptor']}:{initializer['expression']}"
                    for initializer in suggested_descriptor_initializers(r)
                )
                include_changes = "; ".join(
                    f"{change['descriptor']}:{change['action']}"
                    for change in suggested_include_changes(r)
                )
                insertion_point = include_insertion_point(r.declaration_path)
                insertion = ""
                if insertion_point is not None:
                    insertion = (
                        f"line {insertion_point['line']} "
                        f"({insertion_point['strategy']}): {insertion_point['text']}"
                    )
                classification, classification_reasons = recipe_classification(r)
                reasons = "; ".join(classification_reasons)
                print(
                    f"{r.path}\t{r.line}\t{r.risk}\t{r.name}\t{classification}\t{reasons}\t"
                    f"{r.setup_class or ''}\t{r.access_class or ''}\t{r.setup_function or ''}\t"
                    f"{r.access_function or ''}\t{r.legacy_index or ''}\t"
                    f"{r.legacy_alias or ''}\t{r.declaration_path or ''}\t"
                    f"{r.legacy_member_line or ''}\t{r.legacy_member_text or ''}\t"
                    f"{r.constructor_class or ''}\t{r.constructor_function or ''}\t"
                    f"{r.basis_source_line or ''}\t{r.basis_source_text or ''}\t"
                    f"{r.ir_source_line or ''}\t{r.ir_source_text or ''}\t"
                    f"{fields}\t{replacements}\t{initializers}\t{include_changes}\t{insertion}\t{r.hint}\t{r.summary}"
                )
        else:
            print("path\tline\trisk\tcategory\trule\thint\ttext")
            for f in all_findings:
                print(
                    f"{f.path}\t{f.line}\t{f.rule.risk}\t{f.rule.category}\t{f.rule.name}\t"
                    f"{f.rule.hint}\t{f.text}"
                )
    elif args.cache_hazards_only:
        hazard_counts = Counter(h.kind for h in all_cache_hazards)
        print(f"Cache lifecycle hazard candidates: {len(all_cache_hazards)}")
        for kind, count in hazard_counts.most_common():
            print(f"  {kind}: {count}")
        if all_cache_hazards:
            print()
            for hazard in all_cache_hazards:
                for line in format_cache_hazard(hazard):
                    print(line)
    elif args.recipes_only:
        recipes_for_output = [
            recipe for recipe in all_recipes
            if not args.patch_sketches_only or recipe_classification(recipe)[0] == "patch-sketch-candidate"
        ]
        if args.patch_sketches_only:
            print(f"Patch sketch candidates: {len(recipes_for_output)} of {len(all_recipes)} recipe candidate(s)")
        else:
            print(f"Migration recipe candidates: {len(all_recipes)}")
        if recipes_for_output:
            recipe_counts = Counter(r.name for r in recipes_for_output)
            for name, count in recipe_counts.most_common():
                print(f"  {name}: {count}")
            print()
            for recipe in recipes_for_output:
                print(f"{recipe.path}:{recipe.line}: [{recipe.risk}/{recipe.name}] {recipe.summary}")
                print(f"    {recipe.hint}")
                print("    Review classification:")
                for line in format_classification(recipe):
                    print(line)
                legacy_lines = format_legacy_pattern(recipe)
                if legacy_lines:
                    print("    Legacy pattern:")
                    for line in legacy_lines:
                        print(line)
                include_lines = format_include_changes(recipe)
                if include_lines:
                    print("    Suggested include changes:")
                    for line in include_lines:
                        print(line)
                initializer_lines = format_descriptor_initializers(recipe)
                if initializer_lines:
                    print("    Suggested descriptor initializers:")
                    for line in initializer_lines:
                        print(line)
                if recipe.field_counts:
                    print("    Fields used:")
                    for line in format_field_mappings(recipe.field_counts, recipe.phalanx_registration_field_counts):
                        print(line)
                    print("    Suggested local handles:")
                    for line in format_suggested_handles(recipe):
                        print(line)
                    print("    Suggested replacements:")
                    for line in format_replacements(recipe):
                        print(line)
                sketch_lines = format_patch_sketch(recipe)
                if sketch_lines:
                    print("    Non-applied patch sketch:")
                    for line in sketch_lines:
                        print(line)
                print()
    else:
        counts = Counter(f.rule.name for f in all_findings)
        risk_counts = Counter(f.rule.risk for f in all_findings)
        category_counts = Counter(f.rule.category for f in all_findings)
        print(f"Panzer migration advisory scan: {len(all_findings)} finding(s)")
        print("Risk counts: " + ", ".join(f"{k}={risk_counts[k]}" for k in sorted(risk_counts)))
        print("Category counts: " + ", ".join(f"{k}={category_counts[k]}" for k in sorted(category_counts)))
        print("Rule counts:")
        for name, count in counts.most_common():
            print(f"  {name}: {count}")

        grouped: dict[pathlib.Path, list[Finding]] = defaultdict(list)
        for f in all_findings:
            grouped[f.path].append(f)

        print("Top files:")
        for path, findings in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True)[:20]:
            print(f"  {len(findings):4d} {path}")
        print()

        if all_recipes:
            recipe_counts = Counter(r.name for r in all_recipes)
            print(f"Migration recipe candidates: {len(all_recipes)}")
            for name, count in recipe_counts.most_common():
                print(f"  {name}: {count}")
            print("Top recipe candidate files:")
            for recipe in all_recipes[:40]:
                print(f"  {recipe.path}:{recipe.line}: [{recipe.risk}/{recipe.name}] {recipe.summary}")
                print(f"      {recipe.hint}")
                print("      Review classification:")
                for line in format_classification(recipe):
                    print(line)
                legacy_lines = format_legacy_pattern(recipe)
                if legacy_lines:
                    print("      Legacy pattern:")
                    for line in legacy_lines:
                        print(line)
                include_lines = format_include_changes(recipe)
                if include_lines:
                    print("      Suggested include changes:")
                    for line in include_lines:
                        print(line)
                initializer_lines = format_descriptor_initializers(recipe)
                if initializer_lines:
                    print("      Suggested descriptor initializers:")
                    for line in initializer_lines:
                        print(line)
                if recipe.field_counts:
                    print("      Fields used:")
                    for line in format_field_mappings(recipe.field_counts, recipe.phalanx_registration_field_counts):
                        print(line)
                    print("      Suggested local handles:")
                    for line in format_suggested_handles(recipe):
                        print(line)
                    print("      Suggested replacements:")
                    for line in format_replacements(recipe):
                        print(line)
                sketch_lines = format_patch_sketch(recipe)
                if sketch_lines:
                    print("      Non-applied patch sketch:")
                    for line in sketch_lines:
                        print(line)
            print()

        hazard_counts = Counter(h.kind for h in all_cache_hazards)
        print(f"Cache lifecycle hazard candidates: {len(all_cache_hazards)}")
        for kind, count in hazard_counts.most_common():
            print(f"  {kind}: {count}")
        for hazard in all_cache_hazards[:40]:
            for line in format_cache_hazard(hazard):
                print(line)
        print()

        for path, findings in grouped.items():
            print(path)
            for f in findings:
                text = f.text
                if len(text) > args.max_line_length:
                    text = text[: args.max_line_length - 3] + "..."
                print(f"  {f.line}: [{f.rule.risk}/{f.rule.name}] {text}")
                print(f"      {f.rule.hint}")

    return 1 if (
        any(f.rule.risk == "high" for f in all_findings)
        or any(r.risk == "high" for r in all_recipes)
        or any(h.risk == "high" for h in all_cache_hazards)
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
