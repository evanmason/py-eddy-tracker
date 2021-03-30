"""
Normalised Eddy Lifetimes
=========================

"""

import py_eddy_tracker_sample
from matplotlib import pyplot as plt
from numba import njit, prange
from numpy import float32, interp, linspace, unique, where, zeros

from py_eddy_tracker.observations.tracking import TrackEddiesObservations


@njit(cache=True, parallel=True, fastmath=True)
def eddy_norm_lifetime(eddy_var, tracks, xvals, unique_tracks, lifetime_max, out):
    """"""
    for i in prange(unique_tracks.size):
        trk1d_i = where(tracks == unique_tracks[i])[0]
        # out = out + atleast_1d(interp(xvals, linspace(0, 1, trk1d_i.size), eddy_var[trk1d_i]))
        out += interp(xvals, linspace(0, 1, trk1d_i.size), eddy_var[trk1d_i])
    return out / len(unique_tracks)


if __name__ == "__main__":

    plt.close("all")

    a = TrackEddiesObservations.load_file(
        py_eddy_tracker_sample.get_path(
            "eddies_med_adt_allsat_dt2018/Anticyclonic.zarr"
        )
    )
    c = TrackEddiesObservations.load_file(
        py_eddy_tracker_sample.get_path(
          "eddies_med_adt_allsat_dt2018/Cyclonic.zarr"
        )
    )

    unique_tracks, counts = unique(a.tracks, return_counts=True)
    lifetime_max = counts.argmax()
    xvals = linspace(0, 1, lifetime_max)
    AC_radius = zeros(lifetime_max, dtype=float32)
    AC_amplitude = zeros(lifetime_max, dtype=float32)
    CC_radius = zeros(lifetime_max, dtype=float32)
    CC_amplitude = zeros(lifetime_max, dtype=float32)

    # Radius
    AC_radius = eddy_norm_lifetime(
        a.radius_s, a.tracks, xvals, unique_tracks, lifetime_max, AC_radius
    )
    # Amplitude
    AC_amplitude = eddy_norm_lifetime(
        a.amplitude, a.tracks, xvals, unique_tracks, lifetime_max, AC_amplitude
    )

    # Radius
    CC_radius = eddy_norm_lifetime(
        c.radius_s, c.tracks, xvals, unique_tracks, lifetime_max, CC_radius
    )
    # Amplitude
    CC_amplitude = eddy_norm_lifetime(
        c.amplitude, c.tracks, xvals, unique_tracks, lifetime_max, CC_amplitude
    )

    fig, axs = plt.subplots(nrows=2, figsize=(8, 5))

    axs[0].set_title("Normalised Mean Radius")
    axs[0].plot(xvals, AC_radius)
    axs[0].plot(xvals, CC_radius)
    axs[0].set_ylabel("Radius (m)")

    axs[1].set_title("Normalised Mean Amplitude")
    (AA,) = axs[1].plot(xvals, AC_amplitude, label="AC")
    (CC,) = axs[1].plot(xvals, CC_amplitude, label="CC")
    axs[1].set_ylabel("Amplitude (m)")

    axs[1].legend(handles=[AA, CC])

    fig.tight_layout()
    plt.show()
