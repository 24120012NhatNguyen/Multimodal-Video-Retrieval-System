import React, { useState } from "react";
// import { words } from "../helper/words";
import { AiOutlineSearch } from "react-icons/ai";
import { apiHeaders } from "../helper/web_url.js";

function SearchTag({ addTag, web_url, recTags, setRecTags }) {
  const [query, setQuery] = useState("");
  // const [recTags, setRecTags] = useState([]);

  const handleChange = (e) => {
    setQuery(e.target.value);
  };

  const getTypedRec = () => {
    // console.log(query);
    fetch(`${web_url}/getrec`, {
      method: "post",
      headers: apiHeaders(),
      body: JSON.stringify({ text: query }),
    })
      .then((data) => data.json())
      .then((result) => setRecTags(Array.isArray(result) ? result : []))
      .catch((e) => alert("getrec failed!" + e));
  };

  return (
    <div className="relative w-full ">
      <div className="relative w-full">
        <input
          type="search"
          placeholder="Gợi ý lớp object từ mô tả..."
          className="inp inp--sm w-full pr-9"
          onChange={(e) => handleChange(e)}
        ></input>
        <button
          type="button"
          className="absolute right-1 top-1/2 -translate-y-1/2 p-1 rounded text-[color:var(--muted)] hover:text-[color:var(--accent)] transition"
          onClick={() => {
            getTypedRec();
          }}
        >
          <AiOutlineSearch />
        </button>
      </div>
      {recTags.length > 0 && (
        <div className="panel z-20 max-h-[104px] overflow-auto flex flex-wrap gap-1 absolute top-9 p-1.5 w-full">
          {recTags.map((tag) => (
            <span
              key={tag}
              onClick={() => addTag(tag)}
              className="chip cursor-pointer hover:border-[color:var(--accent)] hover:text-[color:var(--accent)]"
            >
              {tag.replace(/_/g, " ")}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export default SearchTag;
