# Gaussian Process

<section>
    <p align="justify">In the most simple term, one can think of Gaussian Process as being distributions over functions. It is a supervised Machine Learning technique and was developed, in the attempt, to solve regression problems. The cool thing about Gaussian Process is that we don't need a "parametric model" to fit the data. It learns using the kernel trick. For an introduction to these techniques, I would recommend the nice review by Prof Zoubin Ghahramani on <a href="http://www.nature.com/nature/journal/v521/n7553/full/nature14541.html">Probabilistic Machine Learning and Artificial Intelligence </a>.</p>




## Examples
Below, we consider two examples: noise-free and a noisy. We show how the Gaussian Process performs well in the noise-free and equally-spaced data as shown below.

<p align="center"><img src="Figures/example_1_uniform.png" alt="uniform_gp" width="60%" height="60%"></p>

<section>
    <p align="justify"> On the other hand, we generate noisy data, which are not equally-spaced. The point which we are mostly interested in is that the Gaussian Process basically demonstrates the level of confidence when we have or do not have data. As expected, we would be more confident when we have more data, and less confident when we don't have data. </p>



<p align="center"><img src="Figures/example_1_non_uniform.png" alt="non_uniform_gp" width="60%" height="60%"></p>

### Further Implementations

<ul>
  <li>Coffee</li>
  <li>Tea</li>
  <li>Milk</li>
</ul>