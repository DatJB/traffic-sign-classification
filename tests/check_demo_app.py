"""Functional UI verification; no model training or batch evaluation."""
import sys
import tempfile
import joblib
from pathlib import Path
from streamlit.testing.v1 import AppTest
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'code'))

app = AppTest.from_file(str(root / 'app.py'), default_timeout=180).run()
assert not app.exception, app.exception
app.radio(key='image_source').set_value('Ảnh mẫu').run()
assert not app.exception, app.exception
assert not app.button(key='classify').disabled
app.button(key='classify').click().run()
assert not app.exception, app.exception
assert app.session_state['prediction']['method'] == 'HOG + kNN'
app.selectbox(key='method').set_value('hsv').run()
assert 'prediction' not in app.session_state.filtered_state
app.button(key='compare').click().run()
assert not app.exception, app.exception
assert len(app.session_state['comparison'][0]) == 3
samples = app.selectbox(key='sample').options
app.selectbox(key='sample').set_value(samples[1]).run()
assert 'comparison' not in app.session_state.filtered_state
assert 'prediction' not in app.session_state.filtered_state
print('Real app: startup, sample prediction, three-model comparison and stale-result clearing passed.', flush=True)

# Exercise actual rendering with a resource-free project, without renaming user files.
source = (root / 'app.py').read_text(encoding='utf-8')
with tempfile.TemporaryDirectory() as directory:
    missing = source.replace('PROJECT_ROOT = Path(__file__).resolve().parent', f'PROJECT_ROOT = Path({directory!r})')
    empty = AppTest.from_string(missing, default_timeout=60).run()
    assert not empty.exception, empty.exception
    assert empty.button(key='classify').disabled and empty.button(key='compare').disabled
    empty.radio(key='image_source').set_value('Ảnh mẫu').run()
    assert not empty.exception and empty.warning
    (Path(directory) / 'models').mkdir()
    joblib.dump({'method': 'HSV + kNN', 'size': [64, 64]}, Path(directory) / 'models/hsv_knn.joblib')
    incomplete = AppTest.from_string(missing, default_timeout=60).run()
    assert not incomplete.exception
    assert incomplete.button(key='classify').disabled
    assert any('chuẩn hóa' in c.value for c in incomplete.caption)
print('Missing models/configuration/samples and absent metrics render safely with prediction disabled.', flush=True)
