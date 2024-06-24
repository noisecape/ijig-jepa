# IJig-Jepa

## Main Idea
This project explores a novel approach to pre-training image models by combining the concept from I-JEPA with a jigsaw puzzle pretext task. Instead of shuffling the image patches themselves, the representations of these patches are shuffled and then encoded by a context encoder. Simultaneously, a target encoder processes the representations in their correct order. The objective is to train the context encoder to produce representations that closely match those of the target encoder, aiming to improve the learning of visual features.

## Future Plans
- **Benchmark Evaluation**: Assess the model's performance on datasets such as ImageNet1K and CIFAR100.
- **Comparative Analysis**: Compare the results with those obtained from the I-JEPA model to validate the effectiveness of the jigsaw puzzle pretext task.
- **Model Refinement**: Continuously refine the model based on evaluation results to enhance its performance and generalization capabilities.
- **Caption Integration**: Explore the use of captions from the Flickr30k dataset to further improve the model's ability to learn and understand visual representations.
