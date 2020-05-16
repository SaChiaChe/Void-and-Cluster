# Void-and-Cluster
A python implementation of the void-and-cluster algorithm

## Description

The void-and-cluster algorithm is a method for generating dither arrays with blue noise characteristic. We implemented the algorithm in python following the paper. Further more, we optimized the process of finding void and clusters and gained 20-30 times speedup for 120x165 dither array generation. The greater the size the greater the speedup is.

We have a jupyter notebook in the notebook folder, check it out if you are interested.

## How to run

```
python VAC.py -i INPUT -o OUTPUT [-s SHAPE SHAPE] [-m MAX_ITER] [-p PROTOTYPE] [-d DITHER] 
    -i INPUT, --input INPUT
        input image path
    -o OUTPUT, --output OUTPUT
        output image path
    -s SHAPE SHAPE, --shape SHAPE SHAPE
        output image shape
    -m MAX_ITER, --max_iter MAX_ITER
        maximum iteration for initial prototype binary pattern generation
    -p PROTOTYPE, --prototype PROTOTYPE
        path to initial prototype binary pattern
    -d DITHER, --dither DITHER
        path to dither matirx
```


example
```
# Start from scratch
python VAC.py -i ./test/test1.jpg -o OUTPUT test1_ht.jpg -s 120 165

# Start the void-and cluster algorithm from a given initial prototype binary pattern
python VAC.py -i ./test/test1.jpg -o OUTPUT test1_ht.jpg -s 120 165 -p ./test/test1_ht.jpg.prototype.png

# Generate halftone image with given dither array
python VAC.py -i ./test/test1.jpg -o OUTPUT test1_ht.jpg -s 120 165 -d ./test/test1_ht.jpg.dither.png
```

## Built With

* Python 3.7.5 :: Anaconda custom (64-bit)

## Authors

* **SaKaTetsu** - *Initial work* - [SaKaTetsu](https://github.com/SaKaTetsu)

## References

[[1]](http://cv.ulichney.com/papers/1993-void-cluster.pdf) Robert A. Ulichney "Void-and-cluster method for dither array generation", Proc. SPIE 1913, Human Vision, Visual Processing, and Digital Display IV, (8 September 1993)