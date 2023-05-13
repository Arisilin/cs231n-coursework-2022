from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dim=np.ndim(np.array(x))
    D=np.prod(x.shape[1:dim])
    xreshape=np.reshape(x,(x.shape[0],D))
    out=xreshape.dot(w)+np.reshape(b,(1,w.shape[1]))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N=x.shape[0]
    dim=len(x.shape)
    D=w.shape[0]
    M=w.shape[1]
    xreshape=np.reshape(x,(N,D))
    dx=dout.dot(w.T) # Strongly recommended for elementwise computation then turn to matrix mul form for dx 
    dx=np.reshape(dx,x.shape)
    # dx=(dout.dot(w.T)).reshape((N)+x.shape[1:dim])
    dw=xreshape.T.dot(dout)# same as previous comment.
    db=np.sum(dout,axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out=np.clip(x,a_min=0,a_max=None)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out=np.clip(x,a_min=0,a_max=None)
    out=(out>0).astype(int)
    dx=dout*out

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N,C=x.shape
    exp_score=np.exp(x)
    rat_true_to_sum=exp_score[range(0,N),y]/np.sum(exp_score,axis=1)
    loss=np.sum(-np.log(rat_true_to_sum))/N
    dx=exp_score/exp_score[range(0,N),y].reshape((N,1)) * rat_true_to_sum.reshape((N,1))
    dx[range(0,N),y]=0
    dx[range(0,N),y]=-np.sum(dx,axis=1)
    dx/=N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        batch_mean=np.mean(x,axis=0)
        batch_var=np.mean((x-batch_mean)**2,axis=0)
        normed_x=(x-batch_mean)/np.sqrt(batch_var+eps)
        out=gamma*normed_x+beta
        cache=(normed_x,gamma,batch_var,eps,batch_mean,x)
        running_mean=running_mean*momentum+batch_mean*(1-momentum)
        running_var=running_var*momentum+batch_var*(1-momentum)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        normed_x=(x-running_mean)/np.sqrt(running_var+eps)
        out=gamma*normed_x+beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    normed_x,gamma,batch_var,eps,batch_mean,x=cache
    N=dout.shape[0]
    dbeta=np.sum(dout,axis=0)
    dgamma=np.sum(normed_x*dout,axis=0)
    # IMPORTANT Attention:gamma should be place inside npsum if you want to reuse this function in layernorm. 
    # since gamma used in transposed way would be turned into vertical one thus it can be no longer take outside of the bracket.
    dx=(dout*gamma-np.mean(dout*gamma,axis=0))/np.sqrt(batch_var+eps)
    dx+=-((batch_var+eps)**(-3/2))/N*(-batch_mean+x)*np.sum(gamma*(x-batch_mean)*dout,axis=0)

    # It's just comparison between some codes...
    # p1a=dout*gamma/np.sqrt(batch_var+eps)
    # p2a=-((batch_var+eps)**(-3/2))/N*(-batch_mean+x)*np.sum(gamma*(x-batch_mean)*dout,axis=0) 
    
    # x_norm,gamma,sample_var,eps,sample_mean,x=cache
    # N = x.shape[0]
    # dbeta = np.sum(dout, axis=0)
    # dgamma = np.sum(x_norm * dout, axis=0)

    # dx_norm = dout * gamma
    # print('copy')
    # dv = ((x - sample_mean) * -0.5 * (sample_var + eps)**-1.5 * dx_norm).sum(axis=0)
    # dm = (dx_norm * -1 * (sample_var + eps)**-0.5).sum(axis=0) + (dv * (x - sample_mean) * -2 / N).sum(axis=0)
    # dx = dx_norm / (sample_var + eps)**0.5 + dv * 2 * (x - sample_mean) / N + dm / N
    # p1b=dx_norm / (sample_var + eps)**0.5
    # p2b= -1 *((x - sample_mean) * dout*gamma ).sum(axis=0) / N * (sample_var + eps)**-1.5* (x - sample_mean) 
    
    # print(p2a)
    # print(p2b)

    # print((dx_norm * -1 * (sample_var + eps)**-0.5).sum(axis=0) /N)
    # print((-np.mean(dout*gamma,axis=0))/np.sqrt(batch_var+eps))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    normed_x,gamma,batch_var,eps,batch_mean,x=cache
    N=dout.shape[0]
    dbeta=np.sum(dout,axis=0)
    dgamma=np.sum(normed_x*dout,axis=0)
    dx=(dout*gamma-np.mean(dout*gamma,axis=0))/np.sqrt(batch_var+eps)
    dx+=-((batch_var+eps)**(-3/2))/N*(-batch_mean+x)*np.sum(gamma*(x-batch_mean)*dout,axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    mean=np.mean(x,axis=1)
    mean=np.expand_dims(mean,axis=1)
    var=np.mean((x-mean)**2,axis=1)
    var=np.expand_dims(var,axis=1)

    normed_x=(x-mean)/np.sqrt(var+eps)
    out=normed_x*gamma+beta
    cache=(mean,var,normed_x,gamma,eps,x)

    # x_T = x.T
    # sample_mean = np.mean(x_T, axis=0)
    # sample_var = np.var(x_T, axis=0)
    # x_norm_T = (x_T - sample_mean) / np.sqrt(sample_var + eps)
    # x_norm = x_norm_T.T
    # out = x_norm * gamma + beta
    # cache = (x, x_norm, gamma, sample_mean, sample_var, eps)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    mean,var,normed_x,gamma,eps,x=cache

    # dx=(dout-np.mean(dout,axis=0))*gamma/np.sqrt(batch_var+eps)
    # dx+=-gamma*((batch_var+eps)**(-3/2))/N*(-batch_mean+x)*np.sum((x-batch_mean)*dout,axis=0)

    dbeta=np.sum(dout,axis=0)
    dgamma=np.sum(normed_x*dout,axis=0)
    
    dx,_,_=batchnorm_backward(dout.T,(normed_x.T,np.expand_dims(gamma,axis=1),var.T,eps,mean.T,x.T))
    dx=dx.T

    # dx=(dout-np.expand_dims(np.mean(dout,axis=1),axis=1))*gamma/np.sqrt(var+eps)
    # dx+=-gamma*((var+eps)**(-3/2))*(-mean+x)*np.expand_dims(np.mean((x-mean)*dout,axis=1),axis=1)

    # sample_mean,sample_var,x_norm,gamma,eps,x=cache

    # x_T = x.T
    # dout_T = dout.T
    # N = x_T.shape[0]
    # dbeta = np.sum(dout, axis=0)
    # dgamma = np.sum(x_norm * dout, axis=0)

    # dx_norm = dout_T * np.expand_dims(gamma,axis=1)
    # sample_var=sample_var.T
    # sample_mean=sample_mean.T
    # print(((x_T - sample_mean) * -0.5 * (sample_var + eps)**-1.5) .shape)
    # dv = ((x_T - sample_mean) * -0.5 * (sample_var + eps)**-1.5 * dx_norm).sum(axis=0)
    # dm = (dx_norm * -1 * (sample_var + eps)**-0.5).sum(axis=0) + (dv * (x_T - sample_mean) * -2 / N).sum(axis=0)
    # dx_T = dx_norm / (sample_var + eps)**0.5 + dv * 2 * (x_T - sample_mean) / N + dm / N
    # dx = dx_T.T    
    # x, x_norm, gamma, sample_mean, sample_var, eps = cache
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask=(np.random.uniform(0,1,size=x.shape)<=p).astype(int)
        out=mask*x/p

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out=x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout * mask/dropout_param["p"]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N,C,H,W=x.shape
    F,_,HH,WW=w.shape
    pad=conv_param['pad']
    stride=conv_param['stride']
    padded_x=np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),mode='constant')
    H2=int(1+(H+2*pad-HH)/stride)
    W2=int(1+(W+2*pad-WW)/stride)
    out=np.zeros((N,F,H2,W2))
    reshaped_w=w.reshape((1,F,C,HH,WW))
    for row in range(0,H2):
      for col in range(0,W2):
        window=padded_x[:,:,row*stride:row*stride+HH,col*stride:col*stride+WW].reshape((N,1,C,HH,WW))
        out[:,:,row,col]=np.sum(window*reshaped_w,axis=(2,3,4))
    out+=b.reshape((1,F,1,1))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x,w,_,conv_param=cache
    N,C,H,W=x.shape
    F,_,HH,WW=w.shape
    pad=conv_param['pad']
    stride=conv_param['stride']
    padded_x=np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),mode='constant')
    
    db=np.sum(dout,axis=(0,2,3))
    
    H2=int(1+(H+2*pad-HH)/stride)
    W2=int(1+(W+2*pad-WW)/stride)
    dx_plain=np.zeros((N,C,H+2*pad,W+2*pad))
    dw=np.zeros_like(w)
    padded_x=np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),mode='constant')
    reshaped_w=w.reshape((1,F,C,HH,WW))
    for row in range(0,H2):
      for col in range(0,W2):
        dx_plain[:,:,row*stride:row*stride+HH,col*stride:col*stride+WW]+=np.sum(dout[:,:,row,col].reshape((N,F,1,1,1))* reshaped_w,axis=1).reshape(N,C,HH,WW)
        dw+=np.sum(dout[:,:,row,col].reshape((N,F,1,1,1))*padded_x[:,:,row*stride:row*stride+HH,col*stride:col*stride+WW].reshape((N,1,C,HH,WW)),axis=0).reshape(F,C,HH,WW)
    dx=dx_plain[:,:,pad:-pad,pad:-pad]
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N,C,H,W=x.shape
    pool_height=pool_param['pool_height']
    pool_width=pool_param['pool_width']
    stride=pool_param['stride']
    
    H2=int(1+(H-pool_height)/stride)
    W2=int(1+(W-pool_width)/stride)

    out=np.zeros((N,C,H2,W2))
    for row in range(H2):
      for col in range(W2):
        out[:,:,row,col]=np.max(x[:,:,row*stride:row*stride+pool_height,col*stride:col*stride+pool_width],axis=(2,3))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x,pool_param=cache
    N,C,H,W=x.shape
    pool_height=pool_param['pool_height']
    pool_width=pool_param['pool_width']
    stride=pool_param['stride']
    dx=np.zeros_like(x)
    H2=int(1+(H-pool_height)/stride)
    W2=int(1+(W-pool_width)/stride)
    out=np.zeros((N,C,H2,W2))
    for row in range(H2):
      for col in range(W2):
        out[:,:,row,col]=np.max(x[:,:,row*stride:row*stride+pool_height,col*stride:col*stride+pool_width],axis=(2,3))

    for row in range(H2):
      for col in range(W2):
        for pos1 in range(pool_height):
          for pos2 in range(pool_width):
            sgn=(x[:,:,row*stride+pos1,col*stride+pos2]==out[:,:,row,col]).astype(int)
            dx[:,:,row*stride+pos1,col*stride+pos2]+=sgn*dout[:,:,row,col]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N,C,H,W=x.shape 
    # x_flattened=x.reshape((N*H*W,C))
    x_flattened=x.transpose(0,2,3,1).reshape((N*H*W,C))# better put your axis to prod together before trying to reshape/stick them.
    out_flattened,cache=batchnorm_forward(x_flattened,gamma,beta,bn_param)
    # out=out_flattened.reshape((N,C,H,W))
    out=out_flattened.reshape((N,H,W,C)).transpose(0,3,1,2)#The same.
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N,C,H,W=dout.shape
    dout_flattened=dout.transpose(0,2,3,1).reshape((N*H*W,C))
    dx_flattened,dgamma,dbeta=batchnorm_backward_alt(dout_flattened,cache)
    dx=dx_flattened.reshape((N,H,W,C)).transpose(0,3,1,2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N,C,H,W=x.shape
    D=int(C/G)
    x_flattened=x.reshape(N,C,H*W)
    gamma_extend=(gamma*np.ones((1,C,H,W))).reshape((C,H,W)) 
    beta_extend=(beta*np.ones((1,C,H,W))).reshape((C,H,W)) 
    out=np.zeros((N,C,H,W))
    cache=[None] * G
    for par in range(G):
      gamma_flatten=(gamma_extend[par*D:par*D+D,:,:]).reshape((D*H*W))
      beta_flatten=(beta_extend[par*D:par*D+D,:,:]).reshape((D*H*W))
      temp,cache[par]=layernorm_forward(x_flattened[:,par*D:par*D+D,:].reshape(N,D*H*W),gamma_flatten,beta_flatten,gn_param)
      out[:,par*D:par*D+D,:,:]=temp.reshape((N,D,H,W))
    cache=dict(cache=cache,G=G)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N,C,H,W=dout.shape
    G=cache['G']
    cache_ind=cache['cache']
    D=int(C/G)
    dout_flattened=dout.reshape(N,C,H*W)
    dx=np.zeros((N,C,H,W))
    dgamma=np.zeros((1,C,1,1))
    dbeta=np.zeros((1,C,1,1))

    for par in range(G):
      dx_flatten,dgamma_flatten,dbeta_flatten=layernorm_backward(dout_flattened[:,par*D:par*D+D,:].reshape(N,D*H*W),cache_ind[par])
      dx[:,par*D:par*D+D,:,:]=dx_flatten.reshape(N,D,H,W)
      dgamma[:,par*D:par*D+D,:,:]=np.sum(dgamma_flatten.reshape(1,D,H,W),axis=(2,3)).reshape((1,D,1,1))
      dbeta[:,par*D:par*D+D,:,:]=np.sum(dbeta_flatten.reshape(1,D,H,W),axis=(2,3)).reshape((1,D,1,1))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
