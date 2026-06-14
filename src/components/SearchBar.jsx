import { useState } from "react";
import "../styles/SearchBar.css";

function SearchBar({ onSearch }) {
  const [query, setQuery] = useState("");

  const handleSearch = () => {
    if (query.trim()) {
      onSearch(query);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  return (
    <div className="search-bar">
      <svg
        width="20"
        height="20"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <circle cx="11" cy="11" r="8"></circle>
        <path d="m21 21-4.35-4.35"></path>
      </svg>

      <input
        type="text"
        placeholder="Search crops, diseases..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyPress={handleKeyPress}
      />
    </div>
  );
}

export default SearchBar;