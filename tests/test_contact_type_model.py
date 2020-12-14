import pytest
from models.contact_type_model import ContactTypeModel


@pytest.mark.integration
def test_predict_model():
    query = 'I called Benjamin for Project Alpha'
    result = ContactTypeModel.predict(query=query)
    assert result is not None
    assert result.get('success')
    assert result.get('data') is not None
    data = result.get('data')
    assert isinstance(data, list)
    entities = data

    for entity in entities:
        assert entity.get('value') is not None
        confidence = entity.get('confidence')
        assert confidence is not None
        assert confidence >= 0
        assert confidence <= 1
        assert entity.get('value') in ['email','call','meeting']


@pytest.mark.integration
def test_predict_model_empty_input():
    query = None
    result = ContactTypeModel.predict(query)
    assert result is not None
    assert result.get('success') is not True
    query = ''
    result = ContactTypeModel.predict(query)
    assert result is not None
    assert result.get('success') is not True

