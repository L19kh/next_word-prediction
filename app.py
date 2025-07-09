
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pickle
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from collections import Counter
import io
import time

# Page configuration
st.set_page_config(
    page_title="🧠 Next Word Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .word-prediction {
        font-size: 1.2rem;
        margin: 0.5rem 0;
        padding: 0.5rem;
        background-color: white;
        border-radius: 5px;
        border: 1px solid #ddd;
    }
    .confidence-bar {
        background-color: #1f77b4;
        height: 20px;
        border-radius: 10px;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

class NextWordPredictor:
    def __init__(self, max_sequence_len=10, vocab_size=10000):
        self.max_sequence_len = max_sequence_len
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.model = None
        self.training_history = None
        
    def preprocess_text(self, text):
        """Clean and preprocess the text data"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\;\:]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def create_sequences(self, text):
        """Create input-output sequences for training"""
        text = self.preprocess_text(text)
        
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts([text])
        
        vocab_size = len(self.tokenizer.word_index) + 1
        sequences = self.tokenizer.texts_to_sequences([text])[0]
        
        input_sequences = []
        for i in range(1, len(sequences)):
            for j in range(1, min(i + 1, self.max_sequence_len + 1)):
                input_sequences.append(sequences[i - j:i + 1])
        
        input_sequences = pad_sequences(input_sequences, maxlen=self.max_sequence_len + 1, padding='pre')
        
        X = input_sequences[:, :-1]
        y = input_sequences[:, -1]
        
        self.vocab_size = min(self.vocab_size, vocab_size)
        y = to_categorical(y, num_classes=self.vocab_size)
        
        return X, y
    
    def build_model(self, embedding_dim=100, lstm_units=128):
        """Build the LSTM model"""
        model = Sequential([
            Embedding(self.vocab_size, embedding_dim, input_length=self.max_sequence_len),
            LSTM(lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2),
            Dense(self.vocab_size, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, text_data, epochs=50, batch_size=64, validation_split=0.2, progress_callback=None):
        """Train the model with progress tracking"""
        X, y = self.create_sequences(text_data)
        self.build_model()
        
        # Custom callback for Streamlit progress
        class StreamlitCallback(tf.keras.callbacks.Callback):
            def __init__(self, progress_bar, status_text):
                self.progress_bar = progress_bar
                self.status_text = status_text
                self.epochs = epochs
            
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / self.epochs
                self.progress_bar.progress(progress)
                self.status_text.text(f'Epoch {epoch + 1}/{self.epochs} - Loss: {logs["loss"]:.4f} - Accuracy: {logs["accuracy"]:.4f}')
        
        callbacks = []
        if progress_callback:
            callbacks.append(progress_callback)
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0,
            callbacks=callbacks
        )
        
        self.training_history = history
        return history
    
    def predict_next_word(self, input_text, top_k=5):
        """Predict the next word given input text"""
        if not self.model or not self.tokenizer:
            return [("Model not trained", 0.0)]
        
        input_text = self.preprocess_text(input_text)
        sequence = self.tokenizer.texts_to_sequences([input_text])[0]
        
        if not sequence:
            return [("the", 0.5), ("a", 0.3), ("and", 0.2)]
        
        sequence = sequence[-self.max_sequence_len:]
        sequence = pad_sequences([sequence], maxlen=self.max_sequence_len, padding='pre')
        
        predictions = self.model.predict(sequence, verbose=0)[0]
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        word_to_index = self.tokenizer.word_index
        index_to_word = {v: k for k, v in word_to_index.items()}
        
        results = []
        for idx in top_indices:
            if idx > 0 and idx in index_to_word:
                word = index_to_word[idx]
                confidence = predictions[idx]
                results.append((word, confidence))
        
        return results

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = NextWordPredictor(max_sequence_len=8, vocab_size=3000)
    st.session_state.model_trained = False
    st.session_state.training_data = ""

# Header
st.markdown('<h1 class="main-header">🧠 Next Word Predictor with LSTM</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Model Configuration")

# Model parameters
max_seq_len = st.sidebar.slider("Max Sequence Length", 5, 20, 8)
vocab_size = st.sidebar.slider("Vocabulary Size", 1000, 5000, 3000)
epochs = st.sidebar.slider("Training Epochs", 10, 100, 30)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)

# Update predictor parameters
st.session_state.predictor.max_sequence_len = max_seq_len
st.session_state.predictor.vocab_size = vocab_size

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📚 Training", "📊 Analysis", "ℹ️ About"])

with tab1:
    st.header("🎯 Next Word Prediction")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Model not trained yet! Please train the model in the Training tab first.")
    else:
        # Input text
        col1, col2 = st.columns([3, 1])
        
        with col1:
            input_text = st.text_input(
                "Enter text to predict next word:",
                value="the quick brown",
                help="Enter some text and get predictions for the next word"
            )
        
        with col2:
            top_k = st.selectbox("Top predictions:", [3, 5, 10], index=1)
        
        if input_text:
            with st.spinner("Predicting..."):
                predictions = st.session_state.predictor.predict_next_word(input_text, top_k=top_k)
            
            st.markdown("### 🎯 Predictions:")
            
            # Display predictions
            for i, (word, confidence) in enumerate(predictions, 1):
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.markdown(f"**{i}. {word}**")
                
                with col2:
                    st.progress(float(confidence))
                    st.caption(f"Confidence: {confidence:.3f}")
            
            # Complete sentence
            if predictions:
                complete_sentence = f"{input_text} {predictions[0][0]}"
                st.markdown("### 💡 Complete Sentence:")
                st.success(f"**{complete_sentence}**")
        
        # Quick test examples
        st.markdown("### 🚀 Quick Examples:")
        example_col1, example_col2, example_col3 = st.columns(3)
        
        examples = [
            "machine learning is",
            "artificial intelligence can",
            "python programming"
        ]
        
        for i, example in enumerate(examples):
            col = [example_col1, example_col2, example_col3][i]
            if col.button(f'"{example}"', key=f"example_{i}"):
                st.rerun()

with tab2:
    st.header("📚 Model Training")
    
    st.info("🎯 **Step 1**: Configure your training data below, then click 'Start Training'")
    
    # Sample data option
    use_sample_data = st.checkbox("Use sample training data", value=True, 
                                   help="Pre-loaded with AI/ML related text for quick testing")
    
    if use_sample_data:
        sample_data = """
        The quick brown fox jumps over the lazy dog. Machine learning is a subset of artificial intelligence.
        Deep learning uses neural networks with multiple layers to analyze complex data patterns.
        Python is a popular programming language for data science and machine learning projects.
        Natural language processing helps computers understand and interpret human language.
        LSTM networks are good at remembering long-term dependencies in sequential data.
        Artificial intelligence can solve complex problems by mimicking human intelligence.
        Data science combines statistics, programming, and domain expertise to extract insights.
        Neural networks are inspired by the human brain and process information through nodes.
        Text preprocessing is essential for natural language processing tasks and data cleaning.
        Tokenization breaks down text into smaller units like words for computational processing.
        Word embeddings represent words as dense vectors in high-dimensional space.
        Recurrent neural networks can process sequential data by maintaining information.
        Gradient descent is an optimization algorithm used to minimize loss functions.
        Overfitting occurs when a model learns training data too well and fails to generalize.
        Cross-validation is a technique used to assess model performance and prevent overfitting.
        """
        training_text = st.text_area("Training Text:", value=sample_data, height=250,
                                    help="Edit this sample data or uncheck the box above to use your own text")
    else:
        training_text = st.text_area("Enter your training text:", height=250,
                                   placeholder="Paste your training text here... The more text you provide, the better the model will perform!")
    
    # Training info
    if training_text:
        word_count = len(training_text.split())
        char_count = len(training_text)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("📝 Words", word_count)
        col2.metric("🔤 Characters", char_count)
        col3.metric("📏 Estimated Training Time", f"{max(1, epochs//10)} min")
        
        if word_count < 50:
            st.warning("⚠️ Very small dataset! Consider adding more text for better results.")
        elif word_count < 200:
            st.info("ℹ️ Small dataset. Model will train quickly but may have limited vocabulary.")
        else:
            st.success("✅ Good dataset size for training!")
    
    st.markdown("---")
    st.markdown("### 🚀 Start Training")
    
    # Training button with enhanced styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        train_button = st.button("🚀 Start Training", disabled=not training_text, 
                                use_container_width=True, type="primary")
    
    if train_button:
        st.session_state.training_data = training_text
        
        # Training progress section
        st.markdown("### 📊 Training Progress")
        
        with st.spinner("Initializing training... This may take a few minutes."):
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_placeholder = st.empty()
            
            # Custom callback for progress
            class StreamlitCallback(tf.keras.callbacks.Callback):
                def __init__(self, progress_bar, status_text, metrics_placeholder, total_epochs):
                    self.progress_bar = progress_bar
                    self.status_text = status_text
                    self.metrics_placeholder = metrics_placeholder
                    self.total_epochs = total_epochs
                
                def on_epoch_end(self, epoch, logs=None):
                    progress = (epoch + 1) / self.total_epochs
                    self.progress_bar.progress(progress)
                    
                    # Status update
                    self.status_text.info(
                        f'🔄 Training Progress: Epoch {epoch + 1}/{self.total_epochs} '
                        f'({progress:.1%} complete)'
                    )
                    
                    # Live metrics
                    with self.metrics_placeholder.container():
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("📉 Loss", f"{logs.get('loss', 0):.4f}")
                        col2.metric("📈 Accuracy", f"{logs.get('accuracy', 0):.4f}")
                        col3.metric("📉 Val Loss", f"{logs.get('val_loss', 0):.4f}")
                        col4.metric("📈 Val Acc", f"{logs.get('val_accuracy', 0):.4f}")
            
            callback = StreamlitCallback(progress_bar, status_text, metrics_placeholder, epochs)
            
            try:
                start_time = time.time()
                
                history = st.session_state.predictor.train(
                    training_text,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    progress_callback=callback
                )
                
                training_time = time.time() - start_time
                
                st.session_state.model_trained = True
                
                # Success message with training summary
                st.success(f"✅ Model trained successfully in {training_time:.1f} seconds!")
                
                # Final metrics display
                st.markdown("### 🎯 Final Training Results")
                final_loss = history.history['loss'][-1]
                final_acc = history.history['accuracy'][-1]
                final_val_loss = history.history['val_loss'][-1]
                final_val_acc = history.history['val_accuracy'][-1]
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("🎯 Training Loss", f"{final_loss:.4f}")
                col2.metric("🎯 Training Accuracy", f"{final_acc:.4f}")
                col3.metric("🔍 Validation Loss", f"{final_val_loss:.4f}")
                col4.metric("🔍 Validation Accuracy", f"{final_val_acc:.4f}")
                
                # Next steps
                st.info("🎉 **Training Complete!** Now go to the 'Prediction' tab to test your model!")
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                st.info("💡 Try reducing the vocabulary size or epochs if you're running into memory issues.")

with tab3:
    st.header("📊 Model Analysis")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Train the model first to see analysis.")
    else:
        # Training history plot
        if st.session_state.predictor.training_history:
            st.subheader("📈 Training History")
            
            history = st.session_state.predictor.training_history.history
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Model Loss', 'Model Accuracy')
            )
            
            # Loss plot
            fig.add_trace(
                go.Scatter(y=history['loss'], name='Training Loss', line=dict(color='red')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(y=history['val_loss'], name='Validation Loss', line=dict(color='orange')),
                row=1, col=1
            )
            
            # Accuracy plot
            fig.add_trace(
                go.Scatter(y=history['accuracy'], name='Training Accuracy', line=dict(color='blue')),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(y=history['val_accuracy'], name='Validation Accuracy', line=dict(color='green')),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        # Vocabulary analysis
        if st.session_state.predictor.tokenizer:
            st.subheader("📚 Vocabulary Analysis")
            
            word_counts = st.session_state.predictor.tokenizer.word_counts
            top_words = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:20])
            
            fig = px.bar(
                x=list(top_words.values()),
                y=list(top_words.keys()),
                orientation='h',
                title='Top 20 Most Frequent Words',
                labels={'x': 'Frequency', 'y': 'Words'}
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Vocabulary stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Vocabulary", len(word_counts))
            col2.metric("Most Frequent Word", max(word_counts, key=word_counts.get))
            col3.metric("Max Frequency", max(word_counts.values()))
        
        # Prediction comparison
        st.subheader("🔍 Prediction Comparison")
        
        sentence_starters = ["the", "machine learning", "artificial intelligence", "python", "deep learning"]
        comparison_data = []
        
        for starter in sentence_starters:
            predictions = st.session_state.predictor.predict_next_word(starter, top_k=3)
            for word, conf in predictions:
                comparison_data.append({
                    'Starter': starter,
                    'Next Word': word,
                    'Confidence': conf
                })
        
        if comparison_data:
            fig = px.bar(
                comparison_data,
                x='Starter',
                y='Confidence',
                color='Next Word',
                title='Next Word Predictions Comparison',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("ℹ️ About This App")
    
    st.markdown("""
    ### 🧠 Next Word Predictor using LSTM
    
    This application uses a **Long Short-Term Memory (LSTM)** neural network to predict the next word in a sequence. 
    
    #### 🔧 How it works:
    1. **Text Preprocessing**: Cleans and tokenizes the input text
    2. **Sequence Creation**: Creates training sequences from the text
    3. **LSTM Training**: Trains a neural network to learn word patterns
    4. **Prediction**: Uses the trained model to predict next words
    
    #### 🏗️ Model Architecture:
    - **Embedding Layer**: Converts words to dense vectors
    - **LSTM Layers**: Two LSTM layers with dropout for regularization
    - **Dense Layer**: Output layer with softmax activation
    
    #### 📊 Features:
    - ✅ Interactive text prediction
    - ✅ Real-time training progress
    - ✅ Visualization of training metrics
    - ✅ Vocabulary analysis
    - ✅ Customizable model parameters
    
    #### 🚀 Usage Tips:
    - Use quality training data for better predictions
    - Adjust sequence length based on your text complexity
    - Monitor training metrics to avoid overfitting
    - Try different vocabulary sizes for optimal performance
    
    ---
    
    **Built with**: Streamlit, TensorFlow, Plotly
    """)
    
    # Model info
    if st.session_state.model_trained:
        st.subheader("🔍 Current Model Info")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Model Parameters:**
            - Max Sequence Length: {st.session_state.predictor.max_sequence_len}
            - Vocabulary Size: {st.session_state.predictor.vocab_size}
            - Total Parameters: {st.session_state.predictor.model.count_params():,}
            """)
        
        with col2:
            if st.session_state.predictor.training_history:
                history = st.session_state.predictor.training_history.history
                st.info(f"""
                **Training Results:**
                - Final Loss: {history['loss'][-1]:.4f}
                - Final Accuracy: {history['accuracy'][-1]:.4f}
                - Validation Loss: {history['val_loss'][-1]:.4f}
                - Validation Accuracy: {history['val_accuracy'][-1]:.4f}
                """)

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666;">🧠 Next Word Predictor - Powered by LSTM Neural Networks</div>',
    unsafe_allow_html=True
)