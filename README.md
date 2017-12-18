# Super resolution methods applied to MRI

This project aims at enhancing the quality of knee MRIs from the Osteoarthritis Initiative using super-resolution methods. The following methods were tested : 
- Spline Interpolation
- Example-based method using [Self similarity and Image priors](https://dl.acm.org/citation.cfm?id=1972664)
- Self Super Resolution ([SSR](https://link.springer.com/chapter/10.1007/978-3-319-46726-9_64))


### Prerequisites

This project was built on Python 3.6.
The following packages are necessary :
```
sklearn, skimage, dipy
```

### Notes
- MRI are 3D images. They have high resolution on the (x,y) and low resolution on the z-axis. Our aim is to enhance the quality on this last axis.
- Although the current implementation of the example-based method works and performs remarkable improvements, it is very time-consuming and needs to optimized for full deployment.
- Our implementation of SSR doesn't perform well on our MRIs. Images fed to the Fourier Burst Accumulation do possess of wide variety of high frequency pattern, and FBA doesn't seem to re-transmit them well in the resulting image. An other limit is that we might not reproduce exactly the MRI recording setting - Rotation matrices applied in the beginning of the algorithm might not be as simple as the one we used (rect or sinc).
