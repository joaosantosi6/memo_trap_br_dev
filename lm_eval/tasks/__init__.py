from pprint import pprint
from typing import List, Union

import lm_eval.base
from . import enem
from . import enem_multimodal
from . import bluex
from . import memo_trap_pt
from . import memo_trap_en
########################################
# All tasks
########################################


TASK_REGISTRY = {
    "memo_trap_pt": memo_trap_pt.MEMO_TRAP_PT,
    "memo_trap_en": memo_trap_en.MEMO_TRAP_EN,

    "enem": enem.ENEM,
    "enem_cot": enem.ENEM_CoT,
    "enem_2022_deprecated": enem.ENEM_2022,
    "enem_cot_2022_deprecated": enem.ENEM_CoT_2022,

    "enem_2022_blind": enem_multimodal.ENEM_2022_BLIND,
    "enem_cot_2022_blind": enem_multimodal.ENEM_CoT_2022_BLIND,
    "enem_2022_images": enem_multimodal.ENEM_2022_IMAGES,
    "enem_cot_2022_images": enem_multimodal.ENEM_CoT_2022_IMAGES,
    "enem_2022_captions": enem_multimodal.ENEM_2022,
    "enem_cot_2022_captions": enem_multimodal.ENEM_CoT_2022,

    "enem_2023_blind": enem_multimodal.ENEM_2023_BLIND,
    "enem_cot_2023_blind": enem_multimodal.ENEM_CoT_2023_BLIND,
    "enem_2023_images": enem_multimodal.ENEM_2023_IMAGES,
    "enem_cot_2023_images": enem_multimodal.ENEM_CoT_2023_IMAGES,
    "enem_2023_captions": enem_multimodal.ENEM_2023,
    "enem_cot_2023_captions": enem_multimodal.ENEM_CoT_2023,

    "bluex_blind": bluex.BLUEX_BLIND,
    "bluex_images": bluex.BLUEX_IMAGES,
    "bluex_captions": bluex.BLUEX_CAPTIONS,
    "bluex_context_captions": bluex.BLUEX_CONTEXT_CAPTIONS,
    "bluex_blind_cot": bluex.BLUEX_BLIND_COT,
    "bluex_images_cot": bluex.BLUEX_IMAGES_COT,
    "bluex_captions_cot": bluex.BLUEX_CAPTIONS_COT,
    "bluex_context_captions_cot": bluex.BLUEX_CONTEXT_CAPTIONS_COT,
}


ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name):
    try:
        return TASK_REGISTRY[task_name]
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")


def get_task_name_from_object(task_object):
    for name, class_ in TASK_REGISTRY.items():
        if class_ is task_object:
            return name

    # this gives a mechanism for non-registered tasks to have a custom name anyways when reporting
    return (
        task_object.EVAL_HARNESS_NAME
        if hasattr(task_object, "EVAL_HARNESS_NAME")
        else type(task_object).__name__
    )


def get_task_dict(task_name_list: List[Union[str, lm_eval.base.Task]]):
    task_name_dict = {
        task_name: get_task(task_name)()
        for task_name in task_name_list
        if isinstance(task_name, str)
    }
    task_name_from_object_dict = {
        get_task_name_from_object(task_object): task_object
        for task_object in task_name_list
        if not isinstance(task_object, str)
    }
    assert set(task_name_dict.keys()).isdisjoint(set(task_name_from_object_dict.keys()))
    return {**task_name_dict, **task_name_from_object_dict}
