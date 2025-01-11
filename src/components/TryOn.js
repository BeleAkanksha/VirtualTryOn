import React from 'react';
import { useParams } from 'react-router-dom';

const TryOn = () => {
  const { productId } = useParams();

  const startTryOn = () => {
    fetch('http://localhost:5000/api/tryon', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ product_id: productId })
    })
      .then(response => response.json())
      .then(data => alert(data.message))
      .catch(error => console.error('Error starting try-on session:', error));
  };

  return (
    <div>
      <h1>Virtual Try-On</h1>
      <button onClick={startTryOn}>Start Try-On</button>
    </div>
  );
};

export default TryOn;
