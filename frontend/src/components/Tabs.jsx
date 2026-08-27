import React from "react";
import { Fragment, useState } from "react";
import { Combobox, Transition } from "@headlessui/react";
import { HiChevronUpDown } from "react-icons/hi2";
import { AiOutlineCheck } from "react-icons/ai";

export default function Tabs({ queryHistory, handleHistory, selected, setSelected }) {
  const [query, setQuery] = useState("");

  const filteredHistory =
    query === ""
      ? queryHistory
      : queryHistory.filter((history) =>
          history.name
            .toLowerCase()
            .replace(/\s+/g, "")
            .includes(query.toLowerCase().replace(/\s+/g, ""))
        );

  return (
    <div className="w-48">
      <Combobox
        value={selected}
        onChange={(e) => {
          setSelected(e);
          console.log("combobox changed!")
          handleHistory(e.id);
        }}
      >
        <div className="relative">
          <div className="relative w-full cursor-default text-left">
            {queryHistory.length > 0 ? (
              <Combobox.Input
                autoComplete="false"
                className="inp inp--sm w-full pl-2.5 pr-8"
                placeholder="Lượt tìm trước..."
                displayValue={(history) => `${history.id}. ${history.name}`}
                onChange={(event) => setQuery(event.target.value)}
              />
            ) : (
              <Combobox.Input
                autoComplete="false"
                className="inp inp--sm w-full pl-2.5 pr-8"
                placeholder="Chưa có lượt tìm nào"
                displayValue={``}
              />
            )}
            <Combobox.Button className="absolute inset-y-0 right-0 flex items-center pr-1.5">
              <HiChevronUpDown className="h-4 w-4 text-[color:var(--muted)]" aria-hidden="true" />
            </Combobox.Button>
          </div>
          <Transition
            as={Fragment}
            enter="transition-all "
            enterFrom="opacity-0 -translate-y-10"
            enterTo="opacity-100 translate-y-0"
            leave="transition-all ease-in-out"
            leaveFrom="opacity-100"
            leaveTo="opacity-0 -translate-y-10	"
            afterLeave={() => setQuery("")}
          >
            <Combobox.Options className="panel z-30 absolute mt-1 max-h-60 w-full overflow-auto py-1 text-[13px] shadow-xl">
              {queryHistory.length === 0 ? (
                <div className="select-none py-2 px-3 text-[color:var(--muted)]">
                  Chưa tìm lượt nào.
                </div>
              ) : filteredHistory.length === 0 && query !== "" ? (
                <div className="select-none py-2 px-3 text-[color:var(--muted)]">
                  Không khớp lượt nào.
                </div>
              ) : (
                filteredHistory.map((history) => (
                  <Combobox.Option
                    key={history.id}
                    className={({ active }) =>
                      `relative cursor-pointer select-none py-1.5 pl-8 pr-3 ${
                        active
                          ? "bg-[color:var(--panel-3)] text-[color:var(--ink)]"
                          : "text-[color:var(--ink-2)]"
                      }`
                    }
                    value={history}
                  >
                    {({ selected, active }) => (
                      <>
                        <span
                          className={`block truncate ${
                            selected ? "font-medium" : "font-normal"
                          }`}
                        >
                          {`${history.id}. ${history.name}`}
                        </span>
                        {selected ? (
                          <span
                            className="absolute inset-y-0 left-0 flex items-center pl-2.5 text-[color:var(--accent)]"
                          >
                            <AiOutlineCheck className="h-4 w-4" aria-hidden="true" />
                          </span>
                        ) : null}
                      </>
                    )}
                  </Combobox.Option>
                ))
              )}
            </Combobox.Options>
          </Transition>
        </div>
      </Combobox>
    </div>
  );
}
