import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { useNavigate } from 'react-router-dom';
import '../styles/Recommendations.css'; // Import the CSS file


const Recommendations = () => {
  const { productId } = useParams();
  const [recommendations, setRecommendations] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    fetch(`http://localhost:5000/api/recommendations/${productId}`)
      .then(response => response.json()
    )
      .then(data => {setRecommendations(data);
      console.log(data);
      })
      .catch(error => console.error('Error fetching recommendations:', error));
  }, [productId]);

  return (
    <div>
      <h1>Recommendations</h1>
      <div className="recommendations-container">
        {recommendations.map(item => (
          <div key={item.id} className="recommendation-card">
            <img src={ `http://localhost:5000/static/${item.image_path}`}/>
            <h2>{item.productDisplayName}</h2>
            <button onClick={() => navigate(`/tryon/${item.id}`)}>Try On</button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Recommendations;