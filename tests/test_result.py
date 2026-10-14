import numpy as np
import pytest
from psm_utils import PSM

from ms2pip.result import ProcessingResult, calculate_correlations


@pytest.mark.parametrize("observed_order", [("b", "y"), ("y", "b")])
def test_calculate_correlations_matches_ion_types(observed_order):
    intensities = {"b": np.array([1.0, 2.0]), "y": np.array([3.0, 4.0])}
    result = ProcessingResult(
        psm_index=0,
        psm=PSM(peptidoform="PEPTIDE/2", spectrum_id="1"),
        predicted_intensity=intensities,
        observed_intensity={ion: intensities[ion] for ion in observed_order},
    )

    calculate_correlations([result])

    assert result.correlation == pytest.approx(1.0)


def test_as_spectra_matches_ion_types_to_mz():
    result = ProcessingResult(
        psm_index=0,
        psm=PSM(peptidoform="PEPTIDE/2", spectrum_id="1"),
        theoretical_mz={"b": np.array([100.0, 200.0]), "y": np.array([150.0, 250.0])},
        predicted_intensity={"y": np.array([3.0, 4.0]), "b": np.array([1.0, 2.0])},
        observed_intensity={"y": np.array([3.0, 4.0]), "b": np.array([1.0, 2.0])},
    )

    predicted, observed = result.as_spectra()

    for spectrum in (predicted, observed):
        np.testing.assert_array_equal(spectrum.mz, [100.0, 150.0, 200.0, 250.0])
        np.testing.assert_array_equal(spectrum.annotations, ["b1", "y1", "b2", "y2"])
        np.testing.assert_allclose(spectrum.intensity, 2 ** np.array([1, 3, 2, 4]) - 0.001)


def test_calculate_correlations_rejects_missing_ion_type():
    result = ProcessingResult(
        psm_index=0,
        psm=PSM(peptidoform="PEPTIDE/2", spectrum_id="1"),
        predicted_intensity={"b": np.array([1.0, 2.0]), "y": np.array([3.0, 4.0])},
        observed_intensity={"b": np.array([1.0, 2.0])},
    )

    with pytest.raises(ValueError, match="Ion types do not match"):
        calculate_correlations([result])


def test_calculate_correlations_rejects_mismatched_ion_count():
    result = ProcessingResult(
        psm_index=0,
        psm=PSM(peptidoform="PEPTIDE/2", spectrum_id="1"),
        predicted_intensity={"b": np.array([1.0, 2.0]), "y": np.array([3.0, 4.0])},
        observed_intensity={"y": np.array([3.0, 4.0]), "b": np.array([1.0])},
    )

    with pytest.raises(ValueError, match="Number of b ions does not match"):
        calculate_correlations([result])
