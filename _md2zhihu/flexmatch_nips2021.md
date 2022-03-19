# 方案动机

文章的出发点是为了解决其他半监督方法比如 FixMatch[2] 中对固定阈值的设置问题，作者 argue 说固定阈值的设置是不合理的，不同类别在不同训练阶段的状态是不一样的，因此文中采用了课程学习（curriculum learning）的思路通过伪标签的状态来自适应的调整半监督训练过程中的阈值，提出的方法叫做 FlexMatch。

# 方法介绍

方法示意图如下所示：

![CPL 示意图](https://cdn.jsdelivr.net/gh/syorami/Paper-Collections@main-md2zhihu-asset/flexmatch_nips2021/4f9f1e39c2125b9d-flexmatch_cpl.png)

半监督训练中的伪标签损失如下所示，其中 <img src="https://www.zhihu.com/equation?tex=%5Ctau" alt="\tau" class="ee_img tr_noresize" eeimg="1"> 为预设的固定阈值：

<img src="https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7B%5Cmu%20B%7D%20%5Csum_%7Bb%3D1%7D%5E%7B%5Cmu%20B%7D%20%5Cmathbb%7B1%7D%5Cleft%28%5Cmax%20%5Cleft%28p_%7Bm%7D%5Cleft%28y%20%5Cmid%20%5Comega%5Cleft%28u_%7Bb%7D%5Cright%29%5Cright%29%5Cright%29%5Cgt%5Ctau%5Cright%29%20H%5Cleft%28%5Chat%7Bp%7D_%7Bm%7D%5Cleft%28y%20%5Cmid%20%5Comega%5Cleft%28u_%7Bb%7D%5Cright%29%5Cright%29%2C%20p_%7Bm%7D%5Cleft%28y%20%5Cmid%20%5Comega%5Cleft%28u_%7Bb%7D%5Cright%29%5Cright%29%5Cright%29%5C%5C" alt="\frac{1}{\mu B} \sum_{b=1}^{\mu B} \mathbb{1}\left(\max \left(p_{m}\left(y \mid \omega\left(u_{b}\right)\right)\right)\gt\tau\right) H\left(\hat{p}_{m}\left(y \mid \omega\left(u_{b}\right)\right), p_{m}\left(y \mid \omega\left(u_{b}\right)\right)\right)\\" class="ee_img tr_noresize" eeimg="1">

FlexMatch 认为某个类别更低的预测准确率表示这个类别的学习效果（learning effect）相对较差，因此需要适当的降低这个类别的阈值，比较直观的解法是通过分割的验证集来判断学习效果，但是由于标注数据本来就比较少了，因此文中提出了采用统计**该类别**预测**置信度高于阈值**的**无标签样本数量**作为学习效果的估计：

<img src="https://www.zhihu.com/equation?tex=%5Csigma_%7Bt%7D%28c%29%3D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%5Cmathbb%7B1%7D%5Cleft%28%5Cmax%20%5Cleft%28p_%7Bm%2C%20t%7D%5Cleft%28y%20%5Cmid%20u_%7Bn%7D%5Cright%29%5Cright%29%5Cgt%5Ctau%5Cright%29%20%5Ccdot%20%5Cmathbb%7B1%7D%5Cleft%28%5Carg%20%5Cmax%20%5Cleft%28p_%7Bm%2C%20t%7D%5Cleft%28y%20%5Cmid%20u_%7Bn%7D%5Cright%29%3Dc%5Cright%29%5Cright.%5C%5C" alt="\sigma_{t}(c)=\sum_{n=1}^{N} \mathbb{1}\left(\max \left(p_{m, t}\left(y \mid u_{n}\right)\right)\gt\tau\right) \cdot \mathbb{1}\left(\arg \max \left(p_{m, t}\left(y \mid u_{n}\right)=c\right)\right.\\" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=%5Cbeta_%7Bt%7D%28c%29%3D%5Cfrac%7B%5Csigma_%7Bt%7D%28c%29%7D%7B%5Cmax%20_%7Bc%7D%20%5Csigma_%7Bt%7D%7D%5C%5C" alt="\beta_{t}(c)=\frac{\sigma_{t}(c)}{\max _{c} \sigma_{t}}\\" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=%5Cmathcal%7BT%7D_%7Bt%7D%28c%29%3D%5Cbeta_%7Bt%7D%28c%29%20%5Ccdot%20%5Ctau%5C%5C" alt="\mathcal{T}_{t}(c)=\beta_{t}(c) \cdot \tau\\" class="ee_img tr_noresize" eeimg="1">

其中， <img src="https://www.zhihu.com/equation?tex=%5Csigma_t%20%28c%29" alt="\sigma_t (c)" class="ee_img tr_noresize" eeimg="1"> 表示类别 <img src="https://www.zhihu.com/equation?tex=c" alt="c" class="ee_img tr_noresize" eeimg="1"> 的学习效果，通过归一化后得到了 <img src="https://www.zhihu.com/equation?tex=%5Cbeta_t%28c%29" alt="\beta_t(c)" class="ee_img tr_noresize" eeimg="1">，其中预测数量最多的类别为 1，然后自适应系数会直接乘到预设的阈值上，来对不同类别调整阈值，最终 <img src="https://www.zhihu.com/equation?tex=%5Ctau_t%28c%29" alt="\tau_t(c)" class="ee_img tr_noresize" eeimg="1"> 为训练计算 loss 时所用的阈值。

考虑到训练前期的时候模型预测不稳定，导致学习效果的估计是有偏的，因此引入了 warmup 过程去解决前期的训练偏差问题，其实就是见过的无标签样本少的时候就让各类别的阈值尽量变小。

<img src="https://www.zhihu.com/equation?tex=%5Cbeta_%7Bt%7D%28c%29%3D%5Cfrac%7B%5Csigma_%7Bt%7D%28c%29%7D%7B%5Cmax%20%5Cleft%5C%7B%5Cmax%20_%7Bc%7D%20%5Csigma_%7Bt%7D%2C%20N-%5Csum_%7Bc%7D%20%5Csigma_%7Bt%7D%5Cright%5C%7D%7D%5C%5C" alt="\beta_{t}(c)=\frac{\sigma_{t}(c)}{\max \left\{\max _{c} \sigma_{t}, N-\sum_{c} \sigma_{t}\right\}}\\" class="ee_img tr_noresize" eeimg="1">

FlexMatch 额外引入了非线性函数来让自适应阈值变得更加平滑，如下所示，采用的非线性函数是 <img src="https://www.zhihu.com/equation?tex=%5Cmathcal%7BM%7D%28x%29%20%3D%20%5Cfrac%7Bx%7D%7B2-x%7D" alt="\mathcal{M}(x) = \frac{x}{2-x}" class="ee_img tr_noresize" eeimg="1">，函数的可视化图形如下图所示（感觉也没啥区别）。

<img src="https://www.zhihu.com/equation?tex=%5Cmathcal%7BT%7D_%7Bt%7D%28c%29%3D%5Cmathcal%7BM%7D%5Cleft%28%5Cbeta_%7Bt%7D%28c%29%5Cright%29%20%5Ccdot%20%5Ctau%5C%5C" alt="\mathcal{T}_{t}(c)=\mathcal{M}\left(\beta_{t}(c)\right) \cdot \tau\\" class="ee_img tr_noresize" eeimg="1">

![非线性映射函数](https://cdn.jsdelivr.net/gh/syorami/Paper-Collections@main-md2zhihu-asset/flexmatch_nips2021/0f61d10ad18ecf37-flexmatch_map_func.png)

整个方法的算法流程如下所示：

![FlexMatch 伪代码](https://cdn.jsdelivr.net/gh/syorami/Paper-Collections@main-md2zhihu-asset/flexmatch_nips2021/4e2626ff8578fda4-flexmatch_alg.png)

# 实验对比

相比于 FixMatch，FlexMatch 在其基础上对半监督的训练性能还是有一定的提升的，同时文中通过实验对比发现 FlexMatch 在收敛速度上更快（阈值降低了一般来说 loss 确实收敛更快）。

![分类实验对比](https://cdn.jsdelivr.net/gh/syorami/Paper-Collections@main-md2zhihu-asset/flexmatch_nips2021/16367b5f74626d63-flexmatch_exp1.png)

部分消融实验对比，warmup 还是有点用的，但是这个映射函数感觉就是作用不是很大，只有不到 0.1% 的提升，实验中还刻意把 y 轴做了调整。

![消融实验对比](https://cdn.jsdelivr.net/gh/syorami/Paper-Collections@main-md2zhihu-asset/flexmatch_nips2021/b63d5c5497f54f2c-flexmatch_exp2.png)

CIFAR-100 上 400 张样本的收敛速度对比实验如下：

![收敛速度对比实验](https://cdn.jsdelivr.net/gh/syorami/Paper-Collections@main-md2zhihu-asset/flexmatch_nips2021/4bc4007ee8c918f5-flexmatch_exp3.png)

# 分析讨论

-   思路很 intuitive，但是我比较好奇的一点是，究竟学习效果差、预测伪标签质量差要减小阈值还是增加阈值，感觉如果是从指标上来说的减小阈值更好一些，可以提高 recall 或者检测中的 mAP，毕竟 precision 的指标作用没有那么大；
-   非线性映射函数的加入感觉没有特别大的必要，不同的映射函数实验区别不大
-   代码开源整合了其他的方法还是挺有价值的，可以帮助后续的半监督方案复现对比

# 参考文献

[1] FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling. NIPS 2021.
[2] FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence. NIPS 2020.
[3] Dash: Semi-Supervised Learning with Dynamic Thresholding. ICML 2021.



Reference:

