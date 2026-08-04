"""
Package to calculate dust extinction and transform between different extinction models
"""

import astropy.units as u
from dust_extinction import parameter_averages
import numpy as np


class DustTools:
    """
    All functionalities for dust extinction and attenuation calculations
    """
    def __int__(self):
        pass


    @staticmethod
    def mag_ext2ebv(mag, wave, ext_law='CCM89', r_v=3.1):
        ext_model = getattr(parameter_averages, ext_law)(Rv=r_v)
        return mag / (ext_model(wave * u.micron) * r_v)

    @staticmethod
    def c00_redd_curve(wavelength=6565, r_v=3.1):
        r"""
        calculate reddening curve
         following  Calzetti et al. (2000) doi:10.1086/308692
         using eq. 4

        :param wavelength: rest frame wavelength in angstrom of spectral part of which to compute the reddening curve
        :type wavelength: float or int
        :param r_v: default 3.1  total extinction at V
        :type r_v: float

        :return extinction E(B - V) in mag
        :rtype: array_like
        """

        # change wavelength from Angstrom to microns
        wavelength *= 1e-4

        # eq. 4
        if (wavelength > 0.63) & (wavelength < 2.20):
            # suitable for 0.63 micron < wavelength < 2.20 micron
            k_lambda = 2.659 * (-1.857 + 1.040/wavelength) + r_v
        elif (wavelength > 0.12) & (wavelength < 0.63):
            # suitable for 0.12 micron < wavelength < 0.63 micron
            k_lambda = 2.659 * (- 2.156 + 1.509 / wavelength - 0.198 / wavelength**2 + 0.011 / wavelength**3) + r_v
        else:
            raise KeyError('wavelength must be > 1200 Angstrom and < 22000 Angstrom')

        return k_lambda

    # @staticmethod
    # def calc_stellar_extinct(wavelength, ebv, r_v):
    #     return ExtinctionTools.compute_reddening_curve(wavelength=wavelength, r_v=r_v) * ebv

    @staticmethod
    def color_ext_ccm89_ebv(wave1, wave2, ebv, r_v=3.1):

        model_ccm89 = parameter_averages.CCM89(Rv=r_v)
        reddening1 = model_ccm89(wave1 * u.micron) * r_v
        reddening2 = model_ccm89(wave2 * u.micron) * r_v

        return (reddening1 - reddening2) * ebv

    @staticmethod
    def band_ext_ccm89_ebv(wave, ebv, r_v=3.1):

        model_ccm89 = parameter_averages.CCM89(Rv=r_v)
        reddening = model_ccm89(wave * u.micron) * r_v

        return reddening * ebv

    @staticmethod
    def ebv2av(ebv, r_v=3.1):
        wave_v = 5388.55 * 1e-4
        model_ccm89 = parameter_averages.CCM89(Rv=r_v)
        return model_ccm89(wave_v*u.micron) * r_v * ebv

    @staticmethod
    def av2ebv(av, r_v=3.1):
        wave_v = 5388.55 * 1e-4
        model_ccm89 = parameter_averages.CCM89(Rv=r_v)
        return av / (model_ccm89(wave_v*u.micron) * r_v)

    @staticmethod
    def color_ext_ccm89_av(wave1, wave2, av, r_v=3.1):

        model_ccm89 = parameter_averages.CCM89(Rv=r_v)
        reddening1 = model_ccm89(wave1*u.micron) * r_v
        reddening2 = model_ccm89(wave2*u.micron) * r_v

        wave_v = 5388.55 * 1e-4
        reddening_v = model_ccm89(wave_v*u.micron) * r_v

        return (reddening1 - reddening2)*av/reddening_v

    @staticmethod
    def color_ext_f99_av(wave1, wave2, av, r_v=3.1):

        model_f99 = parameter_averages.F99(Rv=r_v)
        reddening1 = model_f99(wave1*u.micron) * r_v
        reddening2 = model_f99(wave2*u.micron) * r_v

        wave_v = 5388.55 * 1e-4
        reddening_v = model_f99(wave_v*u.micron) * r_v

        return (reddening1 - reddening2)*av/reddening_v

    @staticmethod
    def compute_balmer_extinction(flux_h_alpha, flux_h_beta):
        """
        Function to compute E(B-V) from H-alpha and H-beta line flux using the Balmer decrement
        following dominguez+13 doi:10.1088/0004-637X/763/2/145 assuming an intrinsic ratio
        H\alpha/H\beta=2.87 (Osterbrock 1989) and the reddening curve from Calzetti et al. (2000) doi:10.1086/308692

        Parameters
        ----------
        flux_h_alpha : float or array-like
        flux_h_beta : float or array-like

        Returns
        -------
        ebv: float or array-like
        """

        # dominguez et al 2013 doi:10.1088/0004-637X/763/2/145
        # eq. 4
        # e_b_v = 1.97 * np.log10(flux_h_alpha / flux_h_beta) - 1.97 * np.log10(2.86)
        e_b_v = 1.97 * np.log10((flux_h_alpha / flux_h_beta) / 2.87 )
        # e_b_v = 1.97 * np.log10((flux_h_alpha / flux_h_beta) / 2.83 )

        return e_b_v

    @staticmethod
    def compute_balmer_extinction_err(flux_h_alpha, flux_h_beta, flux_err_h_alpha, flux_err_h_beta):
        """
        calculate error of colour excess using the Balmer decrement following
        Dominguez et al 2013 doi:10.1088/0004-637X/763/2/145

        Parameters
        ----------
        flux_h_alpha : float or array-like
        flux_h_beta : float or array-like
        flux_err_h_alpha: float or array-like
        flux_err_h_beta : float or array-like

        Returns
        -------
        ebv_err: float or array-like
        """

        # Using error propagation on Dominguez et al 2013 doi:10.1088/0004-637X/763/2/145 eq. 4
        e_b_v_err = np.sqrt((1.97 * flux_err_h_alpha / (flux_h_alpha * np.log(10))) ** 2 +
                            (1.97 * flux_err_h_beta / (flux_h_beta * np.log(10))) ** 2)
        return e_b_v_err

    @staticmethod
    def deredden_flux(flux, wave_mu, ebv, ext_law='G23', r_v=3.1):
        ext_model = getattr(parameter_averages, ext_law)(Rv=r_v)
        return flux / ext_model.extinguish(wave_mu * u.micron, Ebv=ebv)
