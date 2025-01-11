import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import '../styles/Categories.css';

const Categories = () => {
  const [genders, setGenders] = useState([]);

  useEffect(() => {
    axios.get('http://localhost:5000/api/categories')
      .then(response => {
        setGenders(response.data);
        console.log(response.data);
      })
      .catch(error => {
        console.error('There was an error fetching the categories!', error);
      });
  }, []);

  return (
    <div className="categories">
      {genders.length > 0 ? (
        genders.map(gender => (
          <div key={gender} className="category-card">
            <Link className='link' to={`/products/${gender}`}>
              <h2>{gender}</h2>
              {/* <img src={`http://localhost:5000/static/output_images/${gender}_sample.png`} alt={gender} /> */}
            </Link>
          </div>
        ))
      ) : (
        <p>Loading categories...</p>
      )}
    </div>
  );
};

export default Categories;