from src.execution.arb_v1_state_machine import ArbV1State, ArbV1StateMachine


def test_valid_state_transition():
    sm = ArbV1StateMachine()
    result = sm.transition(ArbV1State.DISCOVERED, ArbV1State.PRICED_EXECUTABLE)
    assert result.valid is True


def test_invalid_state_transition():
    sm = ArbV1StateMachine()
    result = sm.transition(ArbV1State.DISCOVERED, ArbV1State.FILLED)
    assert result.valid is False
    assert "invalid transition" in result.reason


def test_partial_fill_must_flatten_before_close():
    sm = ArbV1StateMachine()
    bad = sm.transition(ArbV1State.PARTIAL_FILL, ArbV1State.CLOSED)
    good = sm.transition(ArbV1State.PARTIAL_FILL, ArbV1State.HEDGED_OR_FLATTENED)
    assert bad.valid is False
    assert good.valid is True
