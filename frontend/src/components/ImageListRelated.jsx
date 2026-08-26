import React from "react";

import Image from "next/image";
import { imageUrl } from "../helper/web_url.js";
// Run this in your terminal: npm install react-tooltip

function ImageList({
  image,
}) {
  return (
    <li
      className={`m-0.5 group hover:ease-in-out group duration-300  bg-slate-300 p-0.5 h-max flex relative rounded-md flex relative mb-0.5`}
      key={image}
    >
      <div className="group relative flex h-[169px] w-[300px]">
        <Image
          src={imageUrl(image)}
          // src={"/shoes.jpg"}
          fill={true}
          className="hover:ease-in-out duration-300 relative rounded-md "
        />
      </div>

    </li>
  );
}

export default ImageList;
