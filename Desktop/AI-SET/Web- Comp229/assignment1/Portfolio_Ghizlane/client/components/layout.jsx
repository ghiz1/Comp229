/* 
Ghizlane Ez-Zarrad
301412844
2024-01-28
*/

import React from 'react';
import { Link } from 'react-router-dom';
import football1 from '../src/assets/meG1.jpg';
import '../src/layout.css';

export default function Layout() {
  return (
    <div className="headerContainer">
      <div className="titleContainer">
        <img src={football1} alt="football" className="football" />
        <h1 className="pageTitle">My Portfolio</h1>
      </div>
      <nav className="navLinks">
        <Link to="/" className="navLink boxAnimation">Home</Link> 
        <Link to="/about" className="navLink boxAnimation">About</Link> 
        <Link to="/project" className="navLink boxAnimation">Projects</Link>
        <Link to="/services" className="navLink boxAnimation">Services</Link>  
        <Link to="/contact" className="navLink boxAnimation">Contact me</Link>
      </nav>
      <hr />
    </div>
  );
}
