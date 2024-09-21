"""Run spec functions for lislaam error categorisation."""

from helm.benchmark.adaptation.adapter_spec import (
    ADAPT_GENERATION,
    AdapterSpec,
)
from helm.benchmark.adaptation.common_adapter_specs import (
    get_generation_adapter_spec,
    get_multiple_choice_joint_adapter_spec,
)
from helm.benchmark.metrics.common_metric_specs import (
    get_basic_generation_metric_specs,
    get_classification_metric_specs,
    get_exact_match_metric_specs,
    get_generic_metric_specs,
    get_open_ended_generation_metric_specs,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec

from typing import List, Optional, Dict

@run_spec_function("lislaam_error_cat")
def get_lislaam_error_classification_run_spec() -> RunSpec:
    scenario_spec = ScenarioSpec(class_name="helm.benchmark.scenarios.lislaam_error_cat.LislaamErrorCatScenario")
    adapter_spec = get_generation_adapter_spec(
        instructions="""
        You are a diligent and impartial judge whose task is to carefully assess a [SUMMARY] which contains errors. 
        Given the [ERROR LOCATIONS] you must refer to differences between the [SUMMARY] and [ORIGINAL TEXT] to determine the error type.

        [OUTPUT FORMAT]
        Return 0 if a [SUMMARY] adds details not found in the [ORIGINAL TEXT].
        Else return 1 if the [SUMMARY] mischaracterises [ORIGINAL TEXT] information.
        Give your answer as one integer only.
        """,
        input_noun="Summary",
        output_noun="Error Type",
    )
    metric_specs = get_exact_match_metric_specs() + get_classification_metric_specs() + get_open_ended_generation_metric_specs()
    return RunSpec(
        name="lislaam_error_cat",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["lislaam_error_cat"],
    )
