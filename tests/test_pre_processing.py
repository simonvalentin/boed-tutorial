import pytest
from boed.utils.pre_processing import *


def test_no_change_design():
    desired_design = [[1,1,0], [0,1,0], [1,0,0]]
    user = {"design":[[1,1,0], [0,1,0], [1,0,0]],
           "choices":[[0,0,0], [0,1,2], [1,1,0]],
           "rewards":[[0,0,0], [0,1,0], [1,1,0]]}
    transformed = transform_data(user, desired_design)
    assert transformed["design"] == desired_design, "changed changed even though it should have stayed the same"


def test_no_change_choices():
    desired_design = [[1,1,0], [0,1,0], [1,0,0]]
    user = {"design":[[1,1,0], [0,1,0], [1,0,0]],
           "choices":[[0,0,0], [0,1,2], [1,1,0]],
           "rewards":[[0,0,0], [0,1,0], [1,1,0]]}
    transformed = transform_data(user, desired_design)
    assert transformed["choices"] == [[0,0,0], [0,1,2], [1,1,0]], "data changed even though it should have stayed the same"

    
def test_change_design():
    desired_design = [[1,1,0], [1,0,0], [1,0,0]]
    user = {"design":[[1,1,0], [0,1,0], [1,0,0]],
           "choices":[[0,0,0], [0,1,2], [1,1,0]],
           "rewards":[[0,0,0], [0,1,0], [1,1,0]]}
    transformed = transform_data(user, desired_design)
    assert transformed["design"] == desired_design, "design not changed even though it should have"

    
def test_change_choices():
    desired_design = [[1,1,0], [1,0,0], [1,0,0]]
    user = {"design":[[1,1,0], [0,1,0], [1,0,0]],
           "choices":[[0,0,0], [0,1,2], [1,1,0]],
           "rewards":[[0,0,0], [0,1,0], [1,1,0]]}
    transformed = transform_data(user, desired_design)
    assert transformed["choices"] == [[0,0,0], [1,0,2], [1,1,0]], "data not changed even though it should have"

    
def test_change_blocks_design():
    desired_design = [[1,1,0], [1,0,0], [1,1,0]]
    user = {"design":[[1,1,0], [1,1,0], [1,0,0]],
           "choices":[[0,0,0], [0,1,2], [1,1,0]],
           "rewards":[[0,0,0], [0,1,0], [1,1,0]]}
    transformed = transform_data(user, desired_design)
    assert transformed["design"] == desired_design, "design not changed even though it should have"
    
def test_change_blocks_choices():
    desired_design = [[1,1,0], [1,0,0], [1,1,0]]
    user = {"design":[[1,1,0], [1,1,0], [1,0,0]],
           "choices":[[0,0,0], [0,1,2], [1,1,0]],
           "rewards":[[0,0,0], [0,1,0], [1,1,0]]}
    transformed = transform_data(user, desired_design)
    assert transformed["choices"] == [[0,0,0], [1,1,0], [0,1,2]], "data not changed even though it should have"
    
    
    
def test_extra():
    desired_design = [[0, 0, 0.6], [1, 1, 0]]
    user = {"design":[[1, 0, 1], [0.6, 0, 0]],
           "choices":[[0,1,2], [1,1,0]],
           "rewards":[[0,0,0], [0,1,0]]}
    transformed = transform_data(user, desired_design)
    assert transformed["choices"] == [[0,0,2], [0,2,1]], "data not changed even though it should have"