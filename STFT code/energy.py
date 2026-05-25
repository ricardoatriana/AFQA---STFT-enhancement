import numpy as np
import cv2
from scipy.fftpack import fft2, fftshift, ifft2
from scipy.ndimage import gaussian_filter
import scipy.io as sio

class Energy:
    """    The functions are reimplemented in Python from the matlab implementation by Sharat"""

    @staticmethod
    def raised_cosine_window(BLKSZ, OVRLP):
        """Creates a 2D raised cosine (Hann) window."""
        w = np.hanning(BLKSZ + 2 * OVRLP)[:, None] * np.hanning(BLKSZ + 2 * OVRLP)[None, :]
        return w

    @staticmethod
    def compute_mean_angle(dEnergy, th):
        """Computes the dominant orientation from FFT energy distribution."""
        sth = np.sin(2 * th)
        cth = np.cos(2 * th)
        num = np.sum(dEnergy * sth)
        den = np.sum(dEnergy * cth)
        mth = 0.5 * np.arctan2(num, den)

        # Ensure angles are in [0, π]
        if mth < 0:
            mth += np.pi
        return mth
        
    @staticmethod    
    def compute_mean_frequency(dEnergy, r):
        """Computes dominant ridge frequency using FFT energy distribution."""
        num = np.sum(dEnergy * r)
        den = np.sum(dEnergy)
        return num / (den + np.finfo(float).eps)
    
    @staticmethod
    def compute_coherence(oimg, N=2):
        """Computes the coherence map using Rao's method."""
        h, w = oimg.shape
        cimg = np.zeros((h, w))

        # -----------------
        # Pad the image
        # -----------------
        oimg_padded = np.pad(oimg, ((N, N), (N, N)), mode='reflect')

        # Compute coherence
        for i in range(N, h + N):
            for j in range(N, w + N):
                th = oimg_padded[i, j]
                blk = oimg_padded[i - N:i + N + 1, j - N:j + N + 1]
                cimg[i - N, j - N] = np.sum(np.abs(np.cos(blk - th))) / ((2 * N + 1) ** 2)

        return cimg
        
    @staticmethod
    def get_angular_bw_image(c):
        """Determines the angular bandwidth based on coherence values."""
        bwimg = np.full(c.shape, np.pi / 2)
        bwimg[c <= 0.7] = np.pi
        bwimg[c >= 0.9] = np.pi / 4
        return bwimg
        
    @staticmethod
    def get_angular_filter(t0, bw, angf_pi_4, angf_pi_2, NFFT):
        """Selects the correct precomputed angular filter based on orientation and bandwidth."""
        TSTEPS = angf_pi_4.shape[1]
        DELTAT = np.pi / TSTEPS
        i = int(np.floor((t0 + DELTAT / 2) / DELTAT)) % TSTEPS

        if bw == np.pi / 4:
            return angf_pi_4[:, i].reshape((NFFT, NFFT)).T
        elif bw == np.pi / 2:
            return angf_pi_2[:, i].reshape((NFFT, NFFT)).T
        else:
            return np.ones((NFFT, NFFT))  # Default to no filtering

    def smoothen_orientation_image(oimg):
        # ---------------------------
        # smoothen the image
        # ---------------------------
        gx = np.cos(2 * oimg)
        gy = np.sin(2 * oimg)
        
        # Apply Gaussian filter (equivalent to fspecial and imfilter in MATLAB)
        gfx = gaussian_filter(gx, sigma=1)  # Equivalent to 'symmetric' and 'same'
        gfy = gaussian_filter(gy, sigma=1)
        
        # Calculate the new orientation
        noimg = np.arctan2(gfy, gfx)
        
        # Adjust for negative values
        noimg[noimg < 0] += 2 * np.pi
        
        # Scale the orientation
        noimg *= 0.5
        
        return noimg
    
    @staticmethod
    def compute_energy_and_orientation(img, NFFT, BLKSZ, OVRLP, RMAX, RMIN):
        """Compute energy and orientation maps from FFT."""
        nHt, nWt = img.shape
        img = img.astype(np.float64) / 255  # Normalize image

        nBlkHt = (nHt - 2 * OVRLP) // BLKSZ
        nBlkWt = (nWt - 2 * OVRLP) // BLKSZ
        

        # Compute frequency grid
        x, y = np.meshgrid(np.arange(-NFFT//2, NFFT//2), np.arange(-NFFT//2, NFFT//2))
        r = np.sqrt(x**2 + y**2) + np.finfo(float).eps  # Radial distance

        th = np.arctan2(y, x)
        th[th < 0] += np.pi  # Convert angles to [0, π]
        
        # Bandpass filter setup
        FLOW = NFFT / RMAX
        FHIGH = NFFT / RMIN
        dRLow = 1 / (1 + (r / FHIGH) ** 4)
        dRHigh = 1 / (1 + (FLOW / r) ** 4)
        dBPass = dRLow * dRHigh  # Bandpass filter
        
        # Precompute Angular Filter
        angf_pi_4 = sio.loadmat("/home/ricardoatriana/STFT/angular_filters_pi_4.mat")["angf"]
        angf_pi_2 = sio.loadmat("/home/ricardoatriana/STFT/angular_filters_pi_2.mat")["angf"]

        # Histogram Equalization
        img = cv2.equalizeHist((img * 255).astype(np.uint8))
        
        # Initialize maps
        eimg = np.zeros((nBlkHt, nBlkWt))
        oimg = np.zeros((nBlkHt, nBlkWt))
        fimg = np.zeros((nBlkHt, nBlkWt))
        fftSrc = np.zeros((nBlkHt * nBlkWt, NFFT * NFFT), dtype=np.complex128)

        for i in range(nBlkHt):
            nRow = i * BLKSZ + OVRLP
            for j in range(nBlkWt):
                nCol = j * BLKSZ + OVRLP

                blk = img[nRow - OVRLP:nRow + BLKSZ + OVRLP, nCol - OVRLP:nCol + BLKSZ + OVRLP].astype(np.float64)
                dAvg = np.mean(blk)
                blk -= dAvg  # Remove average intensity
                blk *= Energy.raised_cosine_window(BLKSZ, OVRLP)  # Apply window function

                # Compute FFT (shift spectrum correctly)
                blkfft = fft2(blk, (NFFT, NFFT))
                
                # Compute energy (squared magnitude of FFT)
                dEnergy = np.abs(blkfft) ** 2  
                blkfft *= np.sqrt(dEnergy)  # Normalize FFT by square root of its energy
                
                # Radial Filter bandpass
                blkfft *= dBPass   
                
                # Compute orientation and coherence
                oimg[i, j] = Energy.compute_mean_angle(dEnergy, th)
                oimg_flipped = np.pi - oimg  # Flip the orientation to match the correct ridge flow direction
                cimg = Energy.compute_coherence(oimg_flipped)
                fimg[i, j] = NFFT / (Energy.compute_mean_frequency(dEnergy, r) + np.finfo(float).eps)
                
                # Compute angular bandwidth
                bwimg = Energy.get_angular_bw_image(cimg)
                
                # Apply Angular Filter before storing the FFT coefficients
                af = Energy.get_angular_filter(oimg[i, j], bwimg[i, j], angf_pi_4, angf_pi_2, NFFT)
                blkfft *= af  # Apply angular filter to the FFT block

                # Store the filtered FFT coefficients in fftSrc
                fftSrc[nBlkWt * i + j, :] = blkfft.flatten()
                
                # Store energy (log scale for visualization)
                dTotal = np.sum(dEnergy)  # Total energy for the block
                eimg[i, j] = np.log(dTotal + np.finfo(float).eps)

                    
            
        # DEBUG: Print sample orientation values
        print("Energy sample values:", eimg[:5, :5])
        print("Orientation sample values:", oimg[:5, :5])
        print("Coherence sample values:", cimg[:5, :5])
        print("Frequency sample values:", fimg[:5, :5])

        
        return eimg, oimg_flipped, cimg, fimg, fftSrc, bwimg
        
    def reconstruct_from_fft(fftSrc, NFFT, BLKSZ, OVRLP, img_shape):
        """Reconstruct fingerprint image using stored FFT coefficients."""
        
        # Precompute Angular Filter
        #angf_pi_4 = sio.loadmat("/home/ricardoatriana/STFT/angular_filters_pi_4.mat")["angf"]
        #angf_pi_2 = sio.loadmat("/home/ricardoatriana/STFT/angular_filters_pi_2.mat")["angf"]
        
        nHt, nWt = img_shape
        nBlkHt, nBlkWt = (nHt - 2 * OVRLP) // BLKSZ, (nWt - 2 * OVRLP) // BLKSZ

        # Initialize reconstructed image
        rec_img = np.zeros(img_shape)

        for i in range(nBlkHt):
            nRow = i * BLKSZ + OVRLP
            for j in range(nBlkWt):
                nCol = j * BLKSZ + OVRLP

                blkfft = fftSrc[nBlkWt * i + j, :].reshape(NFFT, NFFT)  # Get FFT block
                 
                blk = np.real(ifft2(blkfft))  # Inverse FFT to get spatial block

                # Overlapping region averaging
                rec_img[nRow-OVRLP:nRow+BLKSZ+OVRLP, nCol-OVRLP:nCol+BLKSZ+OVRLP] += blk[:BLKSZ+2*OVRLP, :BLKSZ+2*OVRLP]

        # Normalize to [0,255] range
        rec_img = np.clip(rec_img, 0, 1)
        return (rec_img * 255).astype(np.uint8)
       