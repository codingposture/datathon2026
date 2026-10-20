import React from 'react';
export default function Icon({ name, className = 'icon' }) {
  return <svg className={className} aria-hidden="true"><use href={`#i-${name}`} /></svg>;
}
