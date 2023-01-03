Maybe I need to modify my PCA pipeline to take into account edge cases where no data is available:

1. Filter out months with too few valid observations
2. Where all values are zero

There is no point on adding these images to the pipeline. But if at prediction time we have images with missing values we will have a hard time evaluating this. We probably need a good strategy for NaNs.

Maybe we can all an image of all zeros and hope for the network to be able to detect this all zero cases yielding no information.

However, currently I already have the PCA values for warm months and they seem to be alright most of the time. So probably its good to just train the network like this.