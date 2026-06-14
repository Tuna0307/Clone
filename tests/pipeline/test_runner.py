import os
import tempfile

import pipeline.runner as r


def test_run_pipeline_basic(mock_llm, mock_embeddings, sample_log_file):
    with tempfile.TemporaryDirectory() as d:
        original_cwd = os.getcwd()
        os.chdir(d)
        try:
            result = r.run_pipeline([sample_log_file])
            assert result is not None
            assert isinstance(result, str)
        finally:
            os.chdir(original_cwd)
