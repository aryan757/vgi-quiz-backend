"""Knowledge-base taxonomy: domain -> topic -> subtopics (Section 5).

Kept separate from script logic so it can be edited without touching code. The seeding
script (Section 7) iterates over this; the topic matcher (Section 8) draws its set of
canonical topic names from here.

Difficulty levels are NOT encoded here — every (domain, topic) is seeded across all three
concrete levels (beginner / intermediate / advanced).
"""

from __future__ import annotations

TAXONOMY: dict[str, dict[str, list[str]]] = {
    "computer_vision": {
        "CNN Fundamentals": ["kernels", "filters", "padding", "stride", "pooling"],
        "Activation Functions": ["relu", "sigmoid", "tanh", "softmax", "leaky relu"],
        "Batch Normalization": ["internal covariate shift", "training stability"],
        "Dropout & Regularization": ["dropout", "l1/l2", "overfitting prevention"],
        "Network Depth Tradeoffs": ["hidden layers", "depth vs width", "vanishing gradients"],
        "Classic CNN Architectures": ["LeNet", "VGG", "ResNet", "Inception"],
        "Object Detection Fundamentals": ["IoU", "anchor boxes", "NMS"],
        "YOLO Family": ["YOLOv5", "YOLOv8", "YOLOv10", "YOLOv11", "YOLO-World", "training/export/inference"],
        "Image Segmentation": ["semantic segmentation", "instance segmentation"],
        "CV Practical Pipeline": ["OpenCV preprocessing", "annotation", "ONNX", "TensorRT", "CoreML", "OpenVINO", "MLflow", "W&B"],
    },
    "machine_learning": {
        "Linear & Logistic Regression": ["linear regression", "logistic regression", "regularization"],
        "Decision Trees": ["splitting criteria", "gini", "entropy", "pruning"],
        "Random Forest": ["bagging", "feature importance", "ensembles"],
        "Bagging & Boosting": ["bagging", "AdaBoost", "gradient boosting", "XGBoost"],
        "KNN": ["distance metrics", "k selection", "curse of dimensionality"],
        "SVM": ["margin", "kernels", "support vectors"],
        "K-Means": ["centroids", "inertia", "elbow method"],
        "DBSCAN": ["density-based clustering", "epsilon", "min samples"],
        "Agglomerative Clustering": ["hierarchical clustering", "linkage", "dendrograms"],
        "Naive Bayes": ["bayes theorem", "conditional independence", "gaussian/multinomial"],
        "Classification vs Regression": ["problem framing", "output types"],
        "Loss Functions": ["MSE", "cross-entropy", "hinge loss"],
        "Evaluation Metrics": ["accuracy", "precision", "recall", "F1", "ROC-AUC", "confusion matrix"],
        "Supervised vs Unsupervised": ["labels", "task types"],
        "Bias-Variance & Overfitting": ["bias-variance tradeoff", "overfitting", "underfitting"],
        "EDA & Distributions": ["exploratory data analysis", "distribution checks", "outliers"],
        "scikit-learn Practical": ["pipelines", "train/test split", "cross-validation", "hyperparameter tuning"],
    },
    "deep_learning": {
        "Neural Network Fundamentals": ["forward pass", "backpropagation intuition", "gradient descent"],
        "RNN": ["sequence modeling", "vanishing gradient", "hidden state"],
        "LSTM": ["forget gate", "input gate", "output gate", "long-term dependencies"],
        "Transformer Architecture": ["self-attention", "multi-head attention", "positional encoding"],
    },
    "genai": {
        "LangChain": ["prompt templates", "chains", "tools", "agents", "memory", "InMemorySaver", "LCEL"],
        "RAG": ["embeddings", "chunking", "vector search", "HNSW", "reranking", "hybrid search"],
        "AI Agents": ["ReAct pattern", "tool calling", "agent loops"],
        "MCP": ["model context protocol", "vs tool call", "why it exists"],
        "Google ADK": ["agent development kit", "concepts", "vs LangChain"],
        "Fine-tuning": ["full fine-tuning", "LoRA", "QLoRA", "prompt-tuning", "instruction-tuning"],
        "Hugging Face": ["Transformers library", "model hub", "pipelines"],
        "Transformers": ["self-attention", "encoder-decoder", "attention is all you need"],
        # Additional well-known interview topics permitted by Section 5.
        "Vector Databases": ["FAISS", "Pinecone", "Chroma", "indexing", "ANN search"],
        "Inference Optimization": ["quantization", "int8", "distillation", "KV cache"],
        "LLM Evaluation": ["benchmarks", "LLM-as-judge", "hallucination metrics"],
    },
}


def all_topics() -> list[str]:
    """Flat list of every canonical topic name across all domains."""
    return [topic for topics in TAXONOMY.values() for topic in topics]


def topic_to_domain() -> dict[str, str]:
    """Map each canonical topic name -> its domain."""
    return {topic: domain for domain, topics in TAXONOMY.items() for topic in topics}


def subtopics_for(topic: str) -> list[str]:
    for topics in TAXONOMY.values():
        if topic in topics:
            return topics[topic]
    return []
