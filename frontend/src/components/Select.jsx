import { Fragment, useState } from "react";
import { Listbox, Transition } from "@headlessui/react";
import { HiChevronUpDown } from "react-icons/hi2";
import { AiOutlineCheck } from "react-icons/ai";

// Hướng lọc theo thời gian trong video.
//
// `value` là thứ backend nhận; `name` chỉ để hiển thị. Trước đây chỗ gọi so sánh
// thẳng với chuỗi hiển thị ("No Filter") để suy ra số — đổi nhãn một chữ là bộ
// lọc hỏng âm thầm.
// Chỉ có tác dụng khi đã bật "Trong kết quả cũ": giữ lại những frame nằm SAU
// (hoặc TRƯỚC) mốc thời gian của kết quả lượt trước, trong cùng video.
export const FILTER_OPTIONS = [
  { value: 0, name: "Cả video" },
  { value: 1, name: "Chỉ cảnh sau" },
  { value: 2, name: "Chỉ cảnh trước" },
];
const people = FILTER_OPTIONS;

export default function Example({selected, setSelected}) {

  return (
    <Listbox
        style={{cursor: 'pointer'}}
        className="w-36 cursor-pointer grow-0"
        value={selected} onChange={(e) => { setSelected(e) }}>
        <div className="relative">
          <Listbox.Button className="inp inp--sm relative w-full pl-2.5 pr-8 text-left">
            <span className="block truncate">{selected.name}</span>
            <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-1.5">
              <HiChevronUpDown className="h-4 w-4 text-[color:var(--muted)]" aria-hidden="true" />
            </span>
          </Listbox.Button>
          <Transition
            as={Fragment}
            leave="transition ease-in duration-100"
            leaveFrom="opacity-100"
            leaveTo="opacity-0"
          >
            <Listbox.Options className="panel z-30 absolute mt-1 max-h-60 w-full overflow-auto py-1 text-[13px] shadow-xl focus:outline-none">
              {people.map((person, personIdx) => (
                <Listbox.Option
                  key={personIdx}
                  className={({ active }) =>
                    `relative select-none cursor-pointer py-1.5 pl-8 pr-3 ${
                      active
                        ? "bg-[color:var(--panel-3)] text-[color:var(--ink)]"
                        : "text-[color:var(--ink-2)]"
                    }`
                  }
                  value={person}
                >
                  {({ selected }) => (
                    <>
                      <span
                        className={`block truncate ${
                          selected ? "font-medium" : "font-normal"
                        }`}
                      >
                        {person.name}
                      </span>
                      {selected ? (
                        <span className="absolute inset-y-0 left-0 flex items-center pl-3 text-amber-600">
                          <AiOutlineCheck
                            className="h-5 w-5"
                            aria-hidden="true"
                          />
                        </span>
                      ) : null}
                    </>
                  )}
                </Listbox.Option>
              ))}
            </Listbox.Options>
          </Transition>
        </div>
      </Listbox>
  );
}
