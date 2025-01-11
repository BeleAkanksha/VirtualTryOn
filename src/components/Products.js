import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import '../styles/Recommendations.css'; // Import the CSS file

const Products = () => {
  const { category } = useParams();
  const [products, setProducts] = useState([]);
  const navigate = useNavigate();


  useEffect(() => {
    axios.get(`http://localhost:5000/api/products/${category}`)
      .then(response => {
        setProducts(response.data);
      })
      .catch(error => {
        console.error('There was an error fetching the products!', error);
      });
  }, [category]);

  return (
    <div className="recommendations-container">
      {products.length > 0 ? (
        products.map(product => (
          <div key={product.id} className="recommendation-card">
            <img src={`http://localhost:5000/static/${product.image_path}`} alt={product.productDisplayName} />
            <h3>{product.productDisplayName}</h3>
            <div className='button-container'>
            <button onClick={() => navigate(`/recommendations/${product.id}`)}>View Recommendations</button>
              <button onClick={() => navigate(`/tryon/${product.id}`)}>Try On</button>
          </div></div>
        ))
      ) : (
        <p>No products found for this category.</p>
      )}
    </div>
  );
};

export default Products;