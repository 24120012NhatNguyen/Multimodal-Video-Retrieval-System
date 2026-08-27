import React from "react";
import LoadingIcon from "./LoadingIcon";

function Questions({ questions, username, setQuestionName, isLoading }) {
  // console.log('rerender')
  return (
    <>
      <div
        key={questions}
        className="panel w-80 top-9 left-0 h-80 z-30 shadow-2xl absolute flex-col gap-1
         content-start p-1"
        id="questions"
        onClick={(e) => {
          e.stopPropagation();
        }}
        style={{
          zIndex: 2,
          display: "none",
        }}
      >
        {isLoading ? (
          <LoadingIcon />
        ) : (
          <>
            <div className="flex flex-wrap overflow-y-auto gap-1 p-1.5 content-start h-40 flex-auto">
              {questions.length > 0 &&
                username &&
                questions
                  .filter((question) => question.owned)
                  .map((question) => {
                    return (
                      <p
                        key={`${question.question}${question.owned}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          document.getElementById("questionName").focus();
                          setQuestionName(e.target.innerText);
                        }}
                        style={{ zIndex: 3 }}
                        className={`chip cursor-pointer break-words hover:border-[color:var(--accent)] hover:text-[color:var(--accent)] 
                      bg-orange-800`}
                      >
                        {question.question}
                      </p>
                    );
                  })}
            </div>
            <div className="flex flex-wrap overflow-y-auto gap-1 p-1.5 content-start h-40 flex-auto">
              {questions.length > 0 &&
                username &&
                questions
                  .filter((question) => !question.owned)
                  .map((question) => {
                    return (
                      <p
                        key={`${question.question}${question.owned}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          document.getElementById("questionName").focus();
                          setQuestionName(e.target.innerText);
                        }}
                        style={{ zIndex: 3 }}
                        className={`chip cursor-pointer hover:border-[color:var(--accent)] hover:text-[color:var(--accent)] 
                      bg-slate-700`}
                      >
                        {question.question}
                      </p>
                    );
                  })}
            </div>
          </>
        )}
      </div>
    </>
  );
}

export default Questions;
