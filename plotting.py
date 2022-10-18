import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 
import pandas as pd


def count_estimate(test_img, test_gt, predict, model):
    N = len(test_gt)
    resume = pd.DataFrame({'GT':[], 'Pred':[], 'Loss':[]})
    for i in range(N):
        IMG = test_img[i,:,:,0]
        GT = test_gt[i,:,:,0]
        PRED = predict[i,:,:,0]
        
        #est_loss = model.get_loss(test_img, test_gt)
        est_count = tf.math.reduce_sum(PRED)
        GT_count = tf.math.reduce_sum(GT)
        
        resume.loc[i, 'GT'] = np.float32(GT_count)
        resume.loc[i, 'Pred'] = np.float32(est_count)
        #resume.loc[i, 'Loss'] = est_loss
        
        
        #est_MAE = model.get_loss(test_img, test_gt)
        plotting_testing_01(IMG, GT, PRED, GT_count, est_count, 0, 0, i) 
    
    resume.to_excel('resume.xlsx')  
               
        
                
def plotting_testing_01(test_img, test_GT, predict, GT_count, est_count, est_loss, est_MAE, i):
    font = {'color':  'black','weight': 'normal','size': 16}

    fig = plt.figure(figsize=(16, 6), constrained_layout=True)
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(test_img, cmap = 'gray')
    ax1.axis('off')
    ax1.set_title('Imagen Real', fontdict=font)
    ax1.text(0.5, -0.1, 
            'Count: %.2f;       Estimate Count: %.2f;    Loss: %.2f;     MAE: %.2f' % (np.float32(GT_count), np.float32(est_count), np.float32(est_loss), np.float32(est_MAE)),
            horizontalalignment='center',
            verticalalignment='top',
            transform=ax1.transAxes, )

    ax2 = fig.add_subplot(1,3,2)
    ax2.set_title('GT', fontdict=font)
    ax2.imshow(test_GT, interpolation='gaussian')
    ax2.axis('off')

    ax3 = fig.add_subplot(1,3,3)
    ax3.set_title('Densidad Estimada', fontdict=font)
    ax3.imshow(predict, interpolation='gaussian')
    ax3.axis('off')


    fig.savefig('plots/test_' + str(i) + '.png');
    plt.close(fig)


