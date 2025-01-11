import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Categories from './components/Categories';
import Products from './components/Products';
import Recommendations from './components/Recommendations';
import TryOn from './components/TryOn';

const App = () => {
  return (
    <Router>
      <Routes>
        <Route exact path="/" element={<Categories />} />
        <Route path="/products/:category" element={<Products />} />
        <Route path="/recommendations/:productId" element={<Recommendations />} />
          <Route path="/tryon/:productId" element={<TryOn />} />
      </Routes>
    </Router>
  );
};

export default App;