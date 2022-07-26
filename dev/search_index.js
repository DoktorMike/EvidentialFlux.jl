var documenterSearchIndex = {"docs":
[{"location":"#EvidentialFlux.jl","page":"EvidentialFlux.jl","title":"EvidentialFlux.jl","text":"","category":"section"},{"location":"","page":"EvidentialFlux.jl","title":"EvidentialFlux.jl","text":"Evidential Deep Learning is a way to generate predictions and the uncertainty associated with them in one single forward pass. This is in stark contrast to traditional Bayesian neural networks which are typically based on Variational Inference, Markov Chain Monte Carlo, Monte Carlo Dropout or Ensembles.","category":"page"},{"location":"#Deep-Evidential-Regression","page":"EvidentialFlux.jl","title":"Deep Evidential Regression","text":"","category":"section"},{"location":"","page":"EvidentialFlux.jl","title":"EvidentialFlux.jl","text":"Deep Evidential Regression[amini2020] is an attempt to apply the principles of Evidential Deep Learning to regression type problems.","category":"page"},{"location":"","page":"EvidentialFlux.jl","title":"EvidentialFlux.jl","text":"It works by putting a prior distribution over the likelihood parameters mathbftheta = mu sigma^2 governing a likelihood model where we observe a dataset mathcalD=x_i y_i_i=1^N where y_i is assumed to be drawn i.i.d. from a Gaussian distribution.","category":"page"},{"location":"","page":"EvidentialFlux.jl","title":"EvidentialFlux.jl","text":"y_i sim mathcalN(mu_i sigma^2_i)","category":"page"},{"location":"","page":"EvidentialFlux.jl","title":"EvidentialFlux.jl","text":"We can express the posterior parameters mathbftheta=mu sigma^2 as p(mathbfthetamathcalD). We seek to create an approximation q(mu sigma^2) = q(mu)(sigma^2) meaning that we assume that the posterior factorizes. This means we can write musimmathcalN(gammasigma^2nu^-1) and sigma^2simGamma^-1(alphabeta). Thus, we can now form","category":"page"},{"location":"","page":"EvidentialFlux.jl","title":"EvidentialFlux.jl","text":"p(mathbfthetamathbfm)=mathcalN(gammasigma^2nu^-1)Gamma^-1(alphabeta)=mathcalN-Gamma^-1(γυαβ)","category":"page"},{"location":"","page":"EvidentialFlux.jl","title":"EvidentialFlux.jl","text":"which can be plugged in to the posterior below.","category":"page"},{"location":"","page":"EvidentialFlux.jl","title":"EvidentialFlux.jl","text":"p(mathbfthetamathbfm y_i) = fracp(y_imathbftheta mathbfm)p(mathbfthetamathbfm)p(y_imathbfm)","category":"page"},{"location":"","page":"EvidentialFlux.jl","title":"EvidentialFlux.jl","text":"Now since the likelihood is Gaussian we would like to put a conjugate prior on the parameters of that likelihood and the Normal Inverse Gamma mathcalN-Gamma^-1(γ υ α β) fits the bill. I'm being a bit handwavy here but this allows us to express the prediction and the associated uncertainty as below.","category":"page"},{"location":"","page":"EvidentialFlux.jl","title":"EvidentialFlux.jl","text":"undersetPredictionunderbracemathbbEmu=gamma\nundersetAleatoricunderbracemathbbEsigma^2=fracbetaalpha-1\nundersetEpistemicunderbracetextVarmu=fracbetanu(alpha-1)","category":"page"},{"location":"","page":"EvidentialFlux.jl","title":"EvidentialFlux.jl","text":"The NIG layer in EvidentialFlux.jl outputs 4 tensors for each target variable, namely gammanualphabeta. This means that in one forward pass we can estimate the prediction, the heteroskedastic aleatoric uncertainty as well as the epistemic uncertainty. Boom!","category":"page"},{"location":"#Theoretical-justifications","page":"EvidentialFlux.jl","title":"Theoretical justifications","text":"","category":"section"},{"location":"","page":"EvidentialFlux.jl","title":"EvidentialFlux.jl","text":"Although for the problems illustrated by Amini et. al., this approach seems to work well it has been shown in [nis2022] that there are theoretical shortcomings regarding the expression of the aleatoric and epistemic uncertainty. They propose a correction of the loss, and the uncertainty calculations. In this package I have implemented both.","category":"page"},{"location":"#Deep-Evidential-Classification","page":"EvidentialFlux.jl","title":"Deep Evidential Classification","text":"","category":"section"},{"location":"","page":"EvidentialFlux.jl","title":"EvidentialFlux.jl","text":"We follow [sensoy2018] in our implementation of Deep Evidential Classification. The neural network layer is implemented to output the alpha_k representing the parameters of a Dirichlet distribution. These parameters has the additional interpretation alpha_k = e_k + 1 where e_k is the evidence for class k. Further, it holds that e_k  0 which is the reason for us modeling them with a softplus activation function. ","category":"page"},{"location":"","page":"EvidentialFlux.jl","title":"EvidentialFlux.jl","text":"Ok, so that's all well and good, but what's the point? Well, the point is that since we are now constructing a network layer that outputs evidence for each class we can apply Dempster-Shafer Theory (DST) to those outputs. DST is a generalization of the Bayesian framework of thought and works by assigning belief mass to states of interest. We can further concretize this notion by Subjective Logic (SL) which places a Dirichlet distribution over these belief masses. Belief masses are defined as b_k=e_kS where e_k is the evidence of state k and S=sum_i^K(e_i+1). Further, SL requires that K+1 states all sum up to 1. This practically means that u+sum_k^Kb_k=1 where u represents the uncertainty of the possible K states, or the \"I don't know.\" class.","category":"page"},{"location":"","page":"EvidentialFlux.jl","title":"EvidentialFlux.jl","text":"Now, since S=sum_i^K(e_i+1)=S=sum_i^K(alpha_i) SL refers to S as the Dirichlet strength which is basically a sum of all the collected evidence in favor of the K outcomes. Consequently the uncertainty u=KS becomes 1 in case there is no evidence available. Therefor, u is a normalized quantity ranging between 0 and 1.","category":"page"},{"location":"#Functions","page":"EvidentialFlux.jl","title":"Functions","text":"","category":"section"},{"location":"","page":"EvidentialFlux.jl","title":"EvidentialFlux.jl","text":"DIR\nNIG\npredict\nuncertainty\naleatoric\nepistemic\nevidence\nnigloss\nnigloss2\ndirloss","category":"page"},{"location":"#EvidentialFlux.DIR","page":"EvidentialFlux.jl","title":"EvidentialFlux.DIR","text":"DIR(in => out; bias=true, init=Flux.glorot_uniform)\nDIR(W::AbstractMatrix, [bias])\n\nA Linear layer with a softplus activation function in the end to implement the Dirichlet evidential distribution. In this layer the number of output nodes should correspond to the number of classes you wish to model. This layer should be used to model a Multinomial likelihood with a Dirichlet prior. Thus the posterior is also a Dirichlet distribution. Moreover the type II maximum likelihood, i.e., the marginal likelihood is a Dirichlet-Multinomial distribution. Create a fully connected layer which implements the Dirichlet Evidential distribution whose forward pass is simply given by:\n\ny = softplus.(W * x .+ bias)\n\nThe input x should be a vector of length in, or batch of vectors represented as an in × N matrix, or any array with size(x,1) == in. The out y will be a vector  of length out, or a batch with size(y) == (out, size(x)[2:end]...) The output will have applied the function softplus(y) to each row/element of y. Keyword bias=false will switch off trainable bias for the layer. The initialisation of the weight matrix is W = init(out, in), calling the function given to keyword init, with default glorot_uniform. The weight matrix and/or the bias vector (of length out) may also be provided explicitly.\n\nArguments:\n\n(in, out): number of input and output neurons\ninit: The function to use to initialise the weight matrix.\nbias: Whether to include a trainable bias vector.\n\n\n\n\n\n","category":"type"},{"location":"#EvidentialFlux.NIG","page":"EvidentialFlux.jl","title":"EvidentialFlux.NIG","text":"NIG(in => out, σ=NNlib.softplus; bias=true, init=Flux.glorot_uniform)\nNIG(W::AbstractMatrix, [bias, σ])\n\nCreate a fully connected layer which implements the NormalInverseGamma Evidential distribution whose forward pass is simply given by:\n\ny = W * x .+ bias\n\nThe input x should be a vector of length in, or batch of vectors represented as an in × N matrix, or any array with size(x,1) == in. The out y will be a vector  of length out*4, or a batch with size(y) == (out*4, size(x)[2:end]...) The output will have applied the function σ(y) to each row/element of y except the first out ones. Keyword bias=false will switch off trainable bias for the layer. The initialisation of the weight matrix is W = init(out*4, in), calling the function given to keyword init, with default glorot_uniform. The weight matrix and/or the bias vector (of length out) may also be provided explicitly. Remember that in this case the number of rows in the weight matrix W MUST be a multiple of 4. The same holds true for the bias vector.\n\nArguments:\n\n(in, out): number of input and output neurons\nσ: The function to use to secure positive only outputs which defaults to the softplus function.\ninit: The function to use to initialise the weight matrix.\nbias: Whether to include a trainable bias vector.\n\n\n\n\n\n","category":"type"},{"location":"#EvidentialFlux.predict","page":"EvidentialFlux.jl","title":"EvidentialFlux.predict","text":"predict(m, x)\n\nReturns the predictions along with the epistemic and aleatoric uncertainty.\n\nArguments:\n\nm: the model which has to have the last layer be Normal Inverse Gamma(NIG) layer\nx: the input data which has to be given as an array or vector\n\n\n\n\n\n","category":"function"},{"location":"#EvidentialFlux.uncertainty","page":"EvidentialFlux.jl","title":"EvidentialFlux.uncertainty","text":"uncertainty(ν, α, β)\n\nCalculates the epistemic uncertainty of the predictions from the Normal Inverse Gamma (NIG) model. Given a textN-Gamma^-1(γ υ α β) distribution we can calculate the epistemic uncertainty as\n\nVarμ = fracβν(α-1)\n\nArguments:\n\nν: the ν parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)\nα: the α parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)\nβ: the β parameter of the NIG distribution which relates to it's uncertainty and whose shape should be (O, B)\n\n\n\n\n\nuncertainty(α, β)\n\nCalculates the aleatoric uncertainty of the predictions from the Normal Inverse Gamma (NIG) model. Given a textN-Gamma^-1(γ υ α β) distribution we can calculate the aleatoric uncertainty as\n\nmathbbEσ^2 = fracβ(α-1)\n\nArguments:\n\nα: the α parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)\nβ: the β parameter of the NIG distribution which relates to it's uncertainty and whose shape should be (O, B)\n\n\n\n\n\nuncertainty(α)\n\nCalculates the epistemic uncertainty associated with a MultinomialDirichlet model (DIR) layer.\n\nα: the α parameter of the Dirichlet distribution which relates to it's concentrations and whose shape should be (O, B)\n\n\n\n\n\n","category":"function"},{"location":"#EvidentialFlux.aleatoric","page":"EvidentialFlux.jl","title":"EvidentialFlux.aleatoric","text":"aleatoric(ν, α, β)\n\nThis is the aleatoric uncertainty as recommended by Meinert, Nis, Jakob Gawlikowski, and Alexander Lavin. 'The Unreasonable Effectiveness of Deep Evidential Regression.' arXiv, May 20, 2022. http://arxiv.org/abs/2205.10060. This is precisely the σ_St from the Student T distribution.\n\nArguments:\n\nν: the ν parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)\nα: the α parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)\nβ: the β parameter of the NIG distribution which relates to it's uncertainty and whose shape should be (O, B)\n\n\n\n\n\n","category":"function"},{"location":"#EvidentialFlux.epistemic","page":"EvidentialFlux.jl","title":"EvidentialFlux.epistemic","text":"epistemic(ν)\n\nThis is the epistemic uncertainty as recommended by Meinert, Nis, Jakob Gawlikowski, and Alexander Lavin. 'The Unreasonable Effectiveness of Deep Evidential Regression.' arXiv, May 20, 2022. http://arxiv.org/abs/2205.10060. \n\nArguments:\n\nν: the ν parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)\n\n\n\n\n\n","category":"function"},{"location":"#EvidentialFlux.evidence","page":"EvidentialFlux.jl","title":"EvidentialFlux.evidence","text":"evidence(α)\n\nCalculates the total evidence of assigning each observation in α to the respective class for a DIR layer.\n\nα: the α parameter of the Dirichlet distribution which relates to it's concentrations and whose shape should be (O, B)\n\n\n\n\n\nevidence(ν, α)\n\nReturns the evidence for the data pushed through the NIG layer. In this setting one way of looking at the NIG distribution is as ν virtual observations governing the mean μ of the likelihood and α virtual observations governing the variance sigma^2. The evidence is then a sum of the virtual observations. Amini et. al. goes through this interpretation in their 2020 paper.\n\nArguments:\n\nν: the ν parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)\nα: the α parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)\n\n\n\n\n\n","category":"function"},{"location":"#EvidentialFlux.nigloss","page":"EvidentialFlux.jl","title":"EvidentialFlux.nigloss","text":"nigloss(y, γ, ν, α, β, λ = 1, ϵ = 0.0001)\n\nThis is the standard loss function for Evidential Inference given a NormalInverseGamma posterior for the parameters of the gaussian likelihood function: μ and σ.\n\nArguments:\n\ny: the targets whose shape should be (O, B)\nγ: the γ parameter of the NIG distribution which corresponds to it's mean and whose shape should be (O, B)\nν: the ν parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)\nα: the α parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)\nβ: the β parameter of the NIG distribution which relates to it's uncertainty and whose shape should be (O, B)\nλ: the weight to put on the regularizer (default: 1)\nϵ: the threshold for the regularizer (default: 0.0001)\n\n\n\n\n\n","category":"function"},{"location":"#EvidentialFlux.nigloss2","page":"EvidentialFlux.jl","title":"EvidentialFlux.nigloss2","text":"nigloss2(y, γ, ν, α, β, λ = 1, p = 1)\n\nThis is the corrected loss function for DER as recommended by Meinert, Nis, Jakob Gawlikowski, and Alexander Lavin. “The Unreasonable Effectiveness of Deep Evidential Regression.” arXiv, May 20, 2022. http://arxiv.org/abs/2205.10060. This is the standard loss function for Evidential Inference given a NormalInverseGamma posterior for the parameters of the gaussian likelihood function: μ and σ.\n\nArguments:\n\ny: the targets whose shape should be (O, B)\nγ: the γ parameter of the NIG distribution which corresponds to it's mean and whose shape should be (O, B)\nν: the ν parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)\nα: the α parameter of the NIG distribution which relates to it's precision and whose shape should be (O, B)\nβ: the β parameter of the NIG distribution which relates to it's uncertainty and whose shape should be (O, B)\nλ: the weight to put on the regularizer (default: 1)\np: the power which to raise the scaled absolute prediction error (default: 1)\n\n\n\n\n\n","category":"function"},{"location":"#EvidentialFlux.dirloss","page":"EvidentialFlux.jl","title":"EvidentialFlux.dirloss","text":"dirloss(y, α, t)\n\nRegularized version of a type II maximum likelihood for the Multinomial(p) distribution where the parameter p, which follows a Dirichlet distribution has been integrated out.\n\nArguments:\n\ny: the targets whose shape should be (O, B)\nα: the parameters of a Dirichlet distribution representing the belief in each class which shape should be (O, B)\nt: counter for the current epoch being evaluated\n\n\n\n\n\n","category":"function"},{"location":"#Index","page":"EvidentialFlux.jl","title":"Index","text":"","category":"section"},{"location":"","page":"EvidentialFlux.jl","title":"EvidentialFlux.jl","text":"","category":"page"},{"location":"#References","page":"EvidentialFlux.jl","title":"References","text":"","category":"section"},{"location":"","page":"EvidentialFlux.jl","title":"EvidentialFlux.jl","text":"[amini2020]: Amini, Alexander, Wilko Schwarting, Ava Soleimany, and Daniela Rus. “Deep Evidential Regression.” ArXiv:1910.02600 [Cs, Stat], November 24, 2020. http://arxiv.org/abs/1910.02600.","category":"page"},{"location":"","page":"EvidentialFlux.jl","title":"EvidentialFlux.jl","text":"[sensoy2018]: Sensoy, Murat, Lance Kaplan, and Melih Kandemir. “Evidential Deep Learning to Quantify Classification Uncertainty.” Advances in Neural Information Processing Systems 31 (June 2018): 3179–89.","category":"page"},{"location":"","page":"EvidentialFlux.jl","title":"EvidentialFlux.jl","text":"[nis2022]: Meinert, Nis, Jakob Gawlikowski, and Alexander Lavin. “The Unreasonable Effectiveness of Deep Evidential Regression.” arXiv, May 20, 2022. http://arxiv.org/abs/2205.10060.","category":"page"}]
}