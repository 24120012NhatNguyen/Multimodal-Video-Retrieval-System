import React from 'react'
import useDragger from "../hooks/useDragger";
import Image from "next/image"

function DragObject({type, id, handleMove}) {

  useDragger(type + id, handleMove)

  return (
    <div
      id={`${type}${id}`}
      title={`${type} — kéo để đặt vị trí, kéo góc để đổi kích thước`}
      className="select-none overflow-hidden resize box top-0 left-0 absolute h-[80px] w-[80px]
                 cursor-move rounded-md border-2 border-[color:var(--accent)]
                 bg-[color:var(--bg)]/40 hover:bg-[color:var(--bg)]/10 transition"
      >
      <Image
        onDragStart={(e) => e.preventDefault()}
        src={`/icons/${type}.png`}
        alt="dragObject"
        layout='fill'
        className=" select-none"
      />
    </div>
  );
}

export default DragObject