from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # loss compute
    num_train=X.shape[0]
    dim=X.shape[1]
    num_class=W.shape[1]
    for i in range(0,num_train):
      softmax_scores_exp=np.exp(X[i].dot(W))
      exp_normed_scores=softmax_scores_exp/np.sum(softmax_scores_exp)
      loss+=-np.log(exp_normed_scores[y[i]])
    loss/=num_train
    loss+= reg*np.sum(W*W)
    #gradient compute
    for i in range(0,num_train):
      softmax_scores_exp=np.exp(X[i].dot(W))
      exp_normed_scores=softmax_scores_exp/np.sum(softmax_scores_exp)# exp_normed_scores[y[i]]=1/g(Wx)=frac
      fraction_of_true_type=exp_normed_scores[y[i]]#
      for j in range(0,num_class): # remeber to multiply 1/g(Wx),g(Wx)=sum exp/ expscoreyi
        if j == y[i]:
          dW[:,j]+=(fraction_of_true_type-1)*X[i]
        else:
          dW[:,j]+=exp_normed_scores[j]*X[i]
    dW/=num_train
    dW+=reg*2*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #vectorized
    num_train=X.shape[0]
    dim=X.shape[1]
    num_class=W.shape[1]
    softmax_scores_exp=np.exp(X.dot(W))
    exp_normed_scores=softmax_scores_exp/np.reshape(np.sum(softmax_scores_exp,axis=1),(num_train,1))
    loss=np.sum(-np.log(exp_normed_scores[range(0,num_train),y]))/num_train
    loss+= reg * np.sum(W * W)
    exp_normed_scores[range(num_train),y]-=1
    dW=(X.T).dot(exp_normed_scores)
    dW/=num_train
    dW+=reg*2*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
