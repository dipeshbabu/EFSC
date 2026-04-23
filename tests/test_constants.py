from efsc.constants import ACTION2ID

def test_actions_present():
    assert ACTION2ID["ANSWER"] == 0
    assert ACTION2ID["REFUSE"] == 3
