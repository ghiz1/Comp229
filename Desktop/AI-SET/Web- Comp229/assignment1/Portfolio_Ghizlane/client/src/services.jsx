/* 
Ghizlane Ez-Zarrad
301412844
2024-01-28
*/
import React from 'react';
import htmlImage from '../src/assets/1.png';
import cssImage from '../src/assets/2.png';
import pythonImage from '../src/assets/3.jpg';
import mlImage from '../src/assets/4.png';
import '../src/services.css';

const Services = () => {
  return (
    <div className="services-container">
      <div className="service">
        <img src={htmlImage} alt="HTML" className="service-image" />
        <p>HTML is the standard markup language for creating web pages and web applications.</p>
      </div>
      <div className="service">
        <img src={cssImage} alt="CSS" className="service-image" />
        <p>CSS is a style sheet language used for describing the presentation of a document written in HTML.</p>
      </div>
      <div className="service">
        <img src={pythonImage} alt="Python" className="service-image" />
        <p>Python is a high-level programming language known for its simplicity and versatility.</p>
      </div>
      <div className="service">
        <img src={mlImage} alt="Machine Learning" className="service-image" />
        <p>Machine learning involves creating algorithms that enable computers to learn from and make predictions based on data.</p>
      </div>
    </div>
  );
};

export default Services;
