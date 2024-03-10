import numpy as np

def levinson_durbin(r, order):
    a = np.zeros(order + 1)
    e = np.zeros(order + 1)

    a[0] = 1.0
    a[1] = -r[1] / r[0]
    e[1] = r[0] + r[1] * a[1]
    
    for k in range(2, order + 1):
        ak = -np.sum([r[j] * a[k - j] for j in range(1, k)]) / e[k - 1]
        a[1:k] += ak * a[1:k][::-1]
        a[k] = ak
        e[k] = e[k - 1] * (1 - ak * ak)
        
    return a

def izracunaj_lpc(audio, red_lpc=20):
    # Autokorelacija
    autocorr = np.correlate(audio, audio, mode='full')
    autocorr = autocorr[len(autocorr)//2:]

    # Levinson-Durbin
    lpc_koeficijenti = levinson_durbin(autocorr, red_lpc)
    
    return lpc_koeficijenti[1:]