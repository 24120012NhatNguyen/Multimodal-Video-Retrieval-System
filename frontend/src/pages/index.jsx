import { useEffect, useRef, useState } from "react";
import React from "react";
import { AiOutlineSearch } from "react-icons/ai";
import Select from "../components/Select.jsx";
import LoadingIcon from "../components/LoadingIcon.jsx";
import ImageListVideo from "../components/ImageListVideo.jsx";
import Panel from "../components/Panel.jsx";
import Tabs from "../components/Tabs.jsx";
import {
  web_url,
  socket_url,
  server,
  session,
  apiHeaders,
} from "../helper/web_url.js";
import VideoWrapper from "../components/VideoWrapper.jsx";
import FullScreen from "../components/FullScreen";
import Questions from "../components/Questions.jsx";
import Lock from "../components/Lock.jsx";
import PageButton from "../components/PageButton.jsx";
import Info from "../components/Info.jsx";
import ExplainBadge from "../components/ExplainBadge.jsx";
import QueryKind from "../components/QueryKind.jsx";
import ChannelModes from "../components/ChannelModes.jsx";
import dynamic from "next/dynamic";
const SpeechToText = dynamic(() => import("../Library/SpeechToText"), {
  ssr: false,
});

let linksArray = [];
let currentK;
let autoFetchData;
const VIDEO_PER_PAGE = 7;
const io = require("socket.io-client");
const socket = io(socket_url, {
  withCredentials: true,
  extraHeaders: {
    "ngrok-skip-browser-warning": "69420",
  },
});

function Index() {
  const [videos, setVideos] = useState({});
  const [id, setId] = useState([]);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState(false);
  const [loading, setLoading] = useState(false);
  const [recTags, setRecTags] = useState([]);
  const [fullScreenImg, setFullScreenImg] = useState(null);
  const [queryHistory, setQueryHistory] = useState([]);
  const [k, setK] = useState(500);
  const [selected, setSelected] = useState(queryHistory[0]);
  const [selectedFilter, setSelectedFilter] = useState({ name: "No Filter" });
  const [relatedObj, setRelatedObj] = useState({});
  const [feedbackMode, setFeedbackMode] = useState(false);
  const [page, setPage] = useState(0);
  const [translate, setTranslate] = useState("");
  const [feedback, setFeedback] = useState({
    lst_pos_idxs: [],
    lst_neg_idxs: [],
  });
  const [questionName, setQuestionName] = useState("");
  const [username, setUsername] = useState("");
  const [lockUsernameInput, setLockUsernameInput] = useState(true);
  const [questions, setQuestions] = useState([]);
  const [questionsLoading, setQuestionsLoading] = useState(false);
  const [rangeFilter, SetRangeFilter] = useState(3);
  const [ignore, setIgnore] = useState(false);
  const [ignoredImages, setIgnoredImages] = useState([]);
  const [autoIgnore, setAutoIgnore] = useState(false);
  const [infoDialog, setInfoDialog] = useState({});
  const [isShown, setIsShown] = useState(false);
  const [searchSpace, setSearchSpace] = useState(0);
  // Ly do moi video noi len: thu hang cua no o tung kenh bang chung.
  const [searchMeta, setSearchMeta] = useState(null);
  const [autofilling, setAutofilling] = useState(false);
  // "auto" = de he thong phan loai; "on"/"off" = nguoi dung ep.
  const [alignMode, setAlignMode] = useState("auto");
  // Công tắc từng kênh do NGƯỜI DÙNG gạt: {asr|ocr|meta|siglip: "on"|"off"}.
  // Thiếu khoá nào thì khoá đó là "auto" (hệ thống tự quyết).
  const [channelModes, setChannelModes] = useState({});
  // Dap an dang co cua cau hoi dang chon -- can de auto-fill giu lai cac dong
  // "manual" thay vi ghi de mat.
  const [submittedInfo, setSubmittedInfo] = useState(null);
  const questionNameRef = useRef("");

  const fetchGetObj = {
    method: "get",
    headers: apiHeaders(),
  };

  // useEffect(() => {
  //  fetch(`${web_url}/data`, fetchGetObj)
  //   .then((data) => data.json())
  //   .then((res) => {
  //    handleData(res);
  //   })
  //   .catch((e) => console.log(`/data fecth error ${e}`));
  // }, []);

  const getOwnedQuestions = (username) => {
    setQuestionsLoading(true);
    fetch(`${socket_url}/getquestions`, {
      method: "post",
      headers: apiHeaders(),
      body: JSON.stringify({
        username: username,
      }),
    })
      .then((res) => res.json())
      .then((res) => {
        // console.log(JSON.stringify(res))
        // console.log(JSON.stringify(questions));
        console.log("set");
        setQuestions(res);
        setQuestionsLoading(false);
      })
      .catch((e) => console.log(e));
  };

  const socketSubmit = (res) => {
    getOwnedQuestions(username);
    if (res && res.data && res.questionName === questionNameRef.current) {
      setSubmittedInfo(res.data);
    }
  };

  useEffect(() => {
    if (!localStorage.getItem("username")) {
      alert("Input username (only first time)");
      document.getElementById("username").focus();
    } else {
      setUsername(localStorage.getItem("username"));
      getOwnedQuestions(localStorage.getItem("username"));
    }

    socket.on("submit", socketSubmit);

    return () => {
      socket.removeAllListeners("submit");
    };
  }, []);

  const socketIgnore = (res) => {
    console.log("on 'ignore'");
    if (questionName === res.questionName) {
      setIgnoredImages(res.data);
    }
  };

  useEffect(() => {
    // let delayInputUsername = setTimeout(() => {
    // Send Axios request here
    // }, 200);
    fetch(`${socket_url}/getignore`, {
      method: "post",
      headers: apiHeaders(),
      body: JSON.stringify({
        questionName: questionName,
      }),
    })
      .then((res) => res.json())
      .then((data) => {
        setIgnoredImages(data.data);
      })
      .catch((e) => console.log(e));

    socket.on("ignore", socketIgnore);

    return () => {
      socket.removeAllListeners("ignore");
    };
  }, [questionName]);

  const getIgnoredImages = (id) => {
    return (ignoredImages || []).includes(id);
  };

  useEffect(() => {
    questionNameRef.current = questionName;
    setSubmittedInfo(null);
    if (questionName) socket.emit("viewsubmitted", { questionName: questionName });
  }, [questionName]);

  useEffect(() => {
    const onView = (res) => {
      if (res && res.data && res.questionName === questionNameRef.current) {
        setSubmittedInfo(res.data);
      }
    };
    socket.on("viewsubmitted", onView);
    socket.on("reorder", onView);
    return () => {
      socket.removeAllListeners("viewsubmitted");
      socket.removeAllListeners("reorder");
    };
  }, []);

  // Chi cac dong nguoi dung tu chon moi duoc giu lai khi auto-fill chay lai.
  const manualAnswers = (submittedInfo ? submittedInfo.lst_idxs || [] : [])
    .map((key, i) => ({
      key: key,
      source: (submittedInfo.lst_sources || [])[i] || "manual",
      video_id: (submittedInfo.lst_video_idxs || [])[i],
      frame_idx: (submittedInfo.lst_keyframe_idxs || [])[i],
    }))
    .filter((e) => e.source === "manual" && e.video_id != null)
    .map((e) => ({
      video_id: e.video_id,
      frame_idx: e.frame_idx,
      source: "manual",
    }));

  useEffect(() => {
    if (username !== "") getOwnedQuestions(username);
  }, [username]);

  //
  //SOCKET.IO
  //

  const handleTranslate = (text) => {
    fetch(`${web_url}/translate`, {
      method: "post",
      headers: apiHeaders(),
      body: JSON.stringify({
        textquery: text,
      }),
    })
      .then((res) => res.json())
      .then((data) => {
        // Endpoint co the tra ve object loi; dat thang object vao state se lam
        // React nem "Objects are not valid as a React child".
        setTranslate(typeof data === "string" ? data : "");
      })
      .catch((e) => console.log(e));
  };

  useEffect(() => {
    if (query !== "") {
      document.getElementById("translate").style.display = "block";
      const mainsearch = document.getElementById("mainsearch");
      mainsearch.scrollLeft = mainsearch.scrollWidth
    }
    const transTime = setTimeout(() => {
      handleTranslate(query);
      // console.log(query);
    }, 350);
    return () => {
      clearTimeout(transTime);
    };
  }, [query]);

  const handleHistory = (id) => {
    setLoading(true);
    currentK = linksArray[id].k;
    setK(linksArray[id].k);
    handleData(linksArray[id].data);
    setLoading(false);
  };

  const handleData = (data) => {
    setPage(0);
    deleteFeedback();
    setVideos(data);
    let ids = [];
    data.forEach((element) => {
      ids = [...ids, ...element.video_info.lst_idxs];
    });
    setId(ids);
  };

  const textSearchFetch = (ignoreIndexes) => {
    let filtervideo =
      selectedFilter.name === "No Filter"
        ? 0
        : selectedFilter.name === "Filter Forwards"
        ? 1
        : 2;
    fetch(`${web_url}/textsearch`, {
      method: "post",
      headers: apiHeaders(),
      body: JSON.stringify({
        // Che do hop nhat cap video: xep hang video bang nhieu kenh bang chung
        // roi moi xuong frame. Duong cu (xep hang frame truc tiep) chay tren
        // dict/ da bi xoa nen khong con dung duoc.
        fusion: true,
        decompose: true,
        // null = de he thong tu quyet dinh; true/false = nguoi dung ep.
        align: alignMode === "auto" ? null : alignMode === "on",
        channel_modes: channelModes,
        textquery: query,
        // Kenh BM25 an tieng Viet; query_en de backend tu phan ra/dich sang
        // tieng Anh cho SigLIP (khong duoc gui tieng Viet vao SigLIP).
        query_vi: query,
        filtervideo: filtervideo,
        filter: filter,
        id: id,
        k: k,
        videos: videos,
        range_filter: rangeFilter,
        ignore: ignore,
        ignore_idxs: ignoreIndexes,
        search_space: searchSpace,
      }),
    })
      .then((data) => data.json())
      .then((raw) => {
        // Che do fusion tra ve object {videos, channels, errors, query};
        // cac duong cu tra ve mang thuan. Chap nhan ca hai.
        const data = Array.isArray(raw) ? raw : raw && raw.videos;
        if (!Array.isArray(data)) {
          const detail = raw && (raw.detail || raw.error);
          const message = Array.isArray(detail)
            ? detail.map((err) => err.msg || JSON.stringify(err)).join("\n")
            : detail || JSON.stringify(raw);
          alert("Textsearch Failed: " + message);
          setLoading(false);
          return;
        }
        if (!Array.isArray(raw)) {
          setSearchMeta({
            channels: raw.channels || {},
            errors: raw.errors || {},
            query: raw.query || null,
            weightsUsed: raw.weights_used || null,
            weightsNote: raw.weights_note || null,
            nRanked: raw.n_videos_ranked || 0,
            aligned: !!raw.aligned,
            alignSkipped: raw.align_skipped || null,
            alignDecidedBy: raw.align_decided_by || null,
          });
        }
        linksArray.push({
          data: data,
          k: k,
        });
        currentK = k;
        setSelected({
          id: queryHistory.length,
          name: query,
        });
        handleData(data);
        setLoading(false);
      })
      .catch((e) => {
        alert("Textsearch Fetch Failed!" + e);
        setLoading(false);
      });
  };

  const getImgLinks = () => {
    let ignoreIndexes;
    setLoading(true);
    setQueryHistory([
      ...queryHistory,
      {
        id: queryHistory.length,
        name: query,
      },
    ]);

    if (ignore) {
      fetch(`${socket_url}/getignore`, {
        method: "post",
        headers: apiHeaders(),
        body: JSON.stringify({
          questionName: questionName,
        }),
      })
        .then((res) => res.json())
        .then((data) => {
          ignoreIndexes = data.data;
          textSearchFetch(ignoreIndexes);
        })
        .catch((e) => {
          alert("Không lấy được danh sách ignore: " + e);
          setLoading(false);
        });
    } else textSearchFetch(ignoreIndexes);
  };

  const clearAll = () => {
    linksArray = [];
    deleteFeedback();
    setQueryHistory([]);
    setVideos([]);
    setId([]);
    setFilter(false);
    setSelectedFilter({ name: "No Filter" });
  };

  const getRec = () => {
    fetch(`${web_url}/getrec`, {
      method: "post",
      headers: apiHeaders(),
      body: JSON.stringify({ text: query }),
    })
      .then((data) => data.json())
      .then((result) => setRecTags(Array.isArray(result) ? result : []))
      .catch((e) => alert("getrec failed!" + e));
  };

  const handleKNN = (imgId) => {
    setLoading(true);
    fetch(`${web_url}/imgsearch?imgid=${imgId}&k=${k}`, fetchGetObj)
      .then((res) => res.json())
      .then((data) => {
        if (!Array.isArray(data)) {
          alert("KNN failed: " + JSON.stringify(data));
          setLoading(false);
          return;
        }
        handleData(data);
        setLoading(false);
      })
      .catch((e) => {
        alert(`KNN Fetch Failed: ${e}`);
        setLoading(false);
      });
  };

  const toggleFullScreen = (image) => {
    if (image !== null) {
      fetch(`${web_url}/relatedimg?imgid=${image.id}`, fetchGetObj)
        .then((res) => res.json())
        .then((data) => {
          setFullScreenImg(image);
          setRelatedObj(data);
        });
    } else setFullScreenImg(null);
  };

  const handleFeedback = (id, type) => {
    setFeedback((oldFeedback) => {
      if (type === "lst_pos_idxs") {
        let lst_pos_idxs;
        if (!oldFeedback.lst_pos_idxs.includes(id)) {
          lst_pos_idxs = [...oldFeedback.lst_pos_idxs, id];
        } else
          lst_pos_idxs = oldFeedback.lst_pos_idxs.filter((item) => item !== id);
        return {
          ...oldFeedback,
          lst_pos_idxs: lst_pos_idxs,
        };
      } else if (type === "lst_neg_idxs") {
        let lst_neg_idxs;
        if (!oldFeedback.lst_neg_idxs.includes(id)) {
          lst_neg_idxs = [...oldFeedback.lst_neg_idxs, id];
        } else
          lst_neg_idxs = oldFeedback.lst_neg_idxs.filter((item) => item !== id);
        return {
          ...oldFeedback,
          lst_neg_idxs: lst_neg_idxs,
        };
      }
    });
  };

  const deleteFeedback = () => {
    setFeedback({
      lst_neg_idxs: [],
      lst_pos_idxs: [],
    });
  };

  const sendFeedback = () => {
    if (
      feedback.lst_neg_idxs.length === 0 &&
      feedback.lst_pos_idxs.length === 0
    ) {
      alert("Feedback first");
      return;
    }
    setLoading(true);
    fetch(`${web_url}/feedback`, {
      method: "post",
      headers: apiHeaders(),
      body: JSON.stringify({
        lst_neg_idxs: feedback.lst_neg_idxs,
        lst_pos_idxs: feedback.lst_pos_idxs,
        k: k,
        videos: videos,
      }),
    })
      .then((res) => res.json())
      .then((data) => {
        if (!Array.isArray(data)) {
          const detail = data && data.detail;
          alert(
            "Feedback failed: " +
              (Array.isArray(detail)
                ? detail.map((err) => err.msg || JSON.stringify(err)).join("\n")
                : detail || JSON.stringify(data))
          );
          setLoading(false);
          return;
        }
        handleData(data);
        setLoading(false);
      })
      .catch((e) => {
        alert("Feedback Fetch Failed! " + e);
        setLoading(false);
      });
  };

  const getImgFeedback = (id) => {
    let imgFeedback;
    if (feedback.lst_pos_idxs.includes(id)) imgFeedback = "like";
    else if (feedback.lst_neg_idxs.includes(id)) imgFeedback = "dislike";
    return imgFeedback;
  };

  const addView = (id, answer = "") => {
    if (questionName === "") {
      alert("Choose question first");
      return;
    }
    if (!socket.connected) {
      alert(
        "Mất kết nối tới socket server (" +
          socket_url +
          "). Đáp án chưa được lưu. Kiểm tra tunnel rồi thử lại."
      );
      return;
    }
    socket.emit("submit", {
      questionName: questionName,
      idx: id,
      user: username,
      answer: answer,
    });
  };

  const handleSelect = (id, video) => {
    if (window.confirm(`Do you want to submit id ${id} in video ${video}?`)) {
      fetch(`${server}?item=${video}&frame=${id}&session=${session}`)
        .then((res) => res.json())
        .then((res) => {
          alert(`Description: ${res.description}. Status: ${res.status}`);
        })
        .catch((e) => alert(e));
    }
  };

  const handleUsername = (name) => {
    localStorage.setItem("username", name);
    setUsername(name);
  };

  const handleIgnore = (lst_idxs) => {
    if (questionName === "") {
      alert("Choose question first");
      return;
    }
    if (!socket.connected) {
      // Không có setLoading ở đây, nhưng nếu socket rớt thì lệnh ignore bị nuốt
      // im lặng và ảnh không bao giờ được đánh dấu -> báo rõ thay vì "treo".
      alert(
        "Mất kết nối tới socket server (" +
          socket_url +
          "). Kiểm tra lại tunnel cloudflared / web_url.js rồi thử lại."
      );
      return;
    }
    socket.emit("ignore", {
      questionName: questionName,
      idx: lst_idxs,
      autoIgnore: false,
    });
  };

  const handleAutoIgnore = (page, isAutoFetched = false) => {
    console.log("isAutofetch: ", isAutoFetched);
    if (questionName === "") {
      alert("Type question first");
      return false;
    } else {
      let lst_video;
      if (isAutoFetched) {
        lst_video =
          videos.length / 7 > 8
            ? videos.slice((Math.floor(videos.length / 7) - 8) * VIDEO_PER_PAGE)
            : videos;
      } else {
        lst_video = videos.slice(
          page * VIDEO_PER_PAGE,
          page * VIDEO_PER_PAGE + VIDEO_PER_PAGE
        );
      }
      let lst_idxs = [];
      lst_video.forEach((video) => {
        if ("video_info_prev" in video) {
          lst_idxs.push(...video.video_info_prev.lst_idxs);
        }
        lst_idxs.push(...video.video_info.lst_idxs);
      });
      // Remember to alert when user forgets to set questions
      // Add autoIgnore in storage
      socket.emit("ignore", {
        questionName: questionName,
        idx: lst_idxs,
        autoIgnore: true,
      });
      return lst_idxs;
    }
  };

  const autoFetch = () => {
    let lst_idxs = handleAutoIgnore(page, true);
    console.log("lstidx ", lst_idxs);
    if (!lst_idxs) {
      return;
    } else {
      console.log(lst_idxs);
      showDialog("success", "Auto Fetching...");
      fetch(`${socket_url}/getignore`, {
        method: "post",
        headers: apiHeaders(),
        body: JSON.stringify({
          questionName: questionName,
        }),
      })
        .then((res) => res.json())
        .then((data) => {
          let ignoreIndexes = data.data;
          console.log("...lst_idxs", ...lst_idxs);
          ignoreIndexes.push(...lst_idxs);
          console.log(ignoreIndexes);

          let filtervideo =
            selectedFilter.name === "No Filter"
              ? 0
              : selectedFilter.name === "Filter Forwards"
              ? 1
              : 2;
          fetch(`${web_url}/textsearch`, {
            method: "post",
            headers: apiHeaders(),
            body: JSON.stringify({
              fusion: true,
              decompose: true,
              align: alignMode === "auto" ? null : alignMode === "on",
              channel_modes: channelModes,
              textquery: query,
              query_vi: query,
              filtervideo: filtervideo,
              filter: filter,
              id: id,
              k: k,
              videos: videos,
              range_filter: rangeFilter,
              ignore: ignore,
              ignore_idxs: ignoreIndexes,
              search_space: searchSpace,
            }),
          })
            .then((data) => data.json())
            .then((raw) => {
              const data = Array.isArray(raw) ? raw : raw && raw.videos;
              if (!Array.isArray(data)) {
                showDialog(
                  "failure",
                  "Auto Fetch Failed: " + JSON.stringify(raw && (raw.error || raw))
                );
                return;
              }
              showDialog("success", "Auto Fetched!");
              setQueryHistory([
                ...queryHistory,
                {
                  id: queryHistory.length,
                  name: query,
                },
              ]);
              linksArray.push({
                data: data,
                k: k,
              });
              currentK = k;

              autoFetchData = data;
            });
        })
        .catch((e) => {
          alert("Auto Fetch Failed!" + e);
        });
    }
  };

  // Viec 3 -- lap day 100 dong. Tinh o app.py (noi co ma tran dac trung) roi
  // day ket qua sang socket_app de luu. Cac dong "manual" giu nguyen o dau.
  const handleAutofill = () => {
    if (questionName === "") {
      alert("Chọn câu hỏi trước");
      return;
    }
    if (!socket.connected) {
      alert("Mất kết nối tới socket server (" + socket_url + ").");
      return;
    }
    const candidates = [];
    (videos || []).forEach((v) => {
      const vi = v.video_info;
      if (!vi) return;
      vi.lst_keyframe_idxs.forEach((fi, i) => {
        candidates.push({
          video_id: v.video_id,
          frame_idx: fi,
          score: vi.lst_scores[i] || 0.5,
        });
      });
    });
    if (candidates.length === 0) {
      alert("Chưa có kết quả tìm kiếm nào để lấp. Tìm kiếm trước đã.");
      return;
    }

    setAutofilling(true);
    fetch(`${socket_url}/getignore`, {
      method: "post",
      headers: apiHeaders(),
      body: JSON.stringify({ questionName: questionName }),
    })
      .then((r) => r.json())
      .then((ig) => {
        const ignore = (ig.data || []).map((key) => {
          const at = String(key).lastIndexOf("#");
          return at < 0
            ? null
            : {
                video_id: String(key).slice(0, at),
                frame_idx: parseInt(String(key).slice(at + 1), 10),
              };
        });
        return fetch(`${web_url}/autofill`, {
          method: "post",
          headers: apiHeaders(),
          body: JSON.stringify({
            manual: manualAnswers,
            candidates: candidates,
            ignore: ignore.filter(Boolean),
            target: 100,
          }),
        });
      })
      .then((r) => r.json())
      .then((res) => {
        if (!res.answers) {
          alert("Auto-fill lỗi: " + JSON.stringify(res));
          setAutofilling(false);
          return;
        }
        socket.emit("setanswers", {
          questionName: questionName,
          user: username,
          answers: res.answers,
        });
        showDialog(
          "success",
          `Đã lấp ${res.n} dòng (${res.n_manual} thủ công, ${res.n_autofill} tự động)`
        );
        setAutofilling(false);
      })
      .catch((e) => {
        alert("Auto-fill lỗi: " + e);
        setAutofilling(false);
      });
  };

  const showDialog = (type, message) => {
    setInfoDialog({ type: type, message: message });
    setIsShown(true);
  };

  const showAutoFetch = () => {
    if (autoFetchData !== undefined) {
      setSelected({
        id: queryHistory.length - 1,
        name: query,
      });
      handleData(autoFetchData);
      autoFetchData = undefined;
    } else {
      showDialog(
        "failure",
        "Fetch hasn't finished! Please wait or manually search!"
      );
    }
  };

  const checkFilter = () => {
    if (filter || (videos.length > 0 && "video_info_prev" in videos[0])) {
      return true;
    }
    return false;
  };

  return (
    <div
      className="flex h-screen w-screen"
      onClick={(e) => {
        document.getElementById("translate").style.display = "none";
        document.getElementById("questions").style.display = "none";
      }}
    >
      {/* {full screen img} */}

      <FullScreen
        fullScreenImg={fullScreenImg}
        setFullScreenImg={setFullScreenImg}
        relatedObj={relatedObj}
      />
      {Object.keys(infoDialog).length !== 0 && isShown && (
        <Info
          type={infoDialog.type}
          message={infoDialog.message}
          setIsShown={setIsShown}
        />
      )}
      <Panel
        socket={socket}
        // handleAutoIgnore={handleAutoIgnore}
        id={id}
        handleKNN={handleKNN}
        recTags={recTags}
        getRec={getRec}
        setRecTags={setRecTags}
        toggleFullScreen={toggleFullScreen}
        handleSelect={handleSelect}
        handleIgnore={handleIgnore}
        ignore={ignore}
        questionName={questionName}
        ignoredImages={ignoredImages}
        getIgnoredImages={getIgnoredImages}
        autoIgnore={autoIgnore}
        searchSpace={searchSpace}
        addView={addView}
      />
      <div className="relative flex-auto h-full flex flex-col overflow-auto">
        {/* {loading icon} */}
        {loading && <LoadingIcon />}
        {/* {searchbars} */}
        <div className="w-full flex flex-col items-center justify-center p-4 bg-white/5 backdrop-blur-md shadow-[0_4px_30px_rgba(0,0,0,0.1)] border border-white/20 rounded-xl mb-4 mt-2 w-[98%] mx-auto z-10">
          <div id="bar" className="w-full flex relative gap-2 items-center">
            {
              // translate &&
              <span
                id="translate"
                onClick={(e) => {
                  e.stopPropagation();
                  navigator.clipboard.writeText(translate);
                  document.getElementById("mainsearch").focus();
                }}
                style={{ zIndex: 2, display: "none" }}
                className="hover:ring-2 ring-orange-400 transition-all cursor-pointer align-middle h-fit absolute top-14 placeholder:italic text-slate-300 w-full p-2 indent-2 rounded-md bg-slate-800/90 backdrop-blur-sm"
              >
                {translate ? translate : "Translate..."}
              </span>
            }
            <input
              tabIndex={-1}
              id="K"
              type="number"
              placeholder="K"
              className="w-16 h-12 transition-all hover:drop-shadow-[0px_4px_10px_rgba(255,255,255,0.2)] placeholder:italic text-slate-300 indent-2 text-lg relative rounded-full bg-slate-900/50 border border-slate-700 focus:outline-none focus:ring-2 focus:ring-orange-500"
              onChange={(e) => {
                const val = e.target.value === "" ? 500 : Number(e.target.value);
                const parsed = Number.isNaN(val) ? 500 : val;
                if (filter && parsed > currentK)
                  alert(
                    `Filter Mode: K must be smaller than in the previous query`
                  );
                else setK(parsed);
              }}
              value={k}
            ></input>
            <input
              id="mainsearch"
              tabIndex={1}
              autoFocus={true}
              onKeyDown={(e) => {
                if (e.key == "Enter") {
                  document.getElementById("mainsearch_button").click();
                }
              }}
              type="search"
              placeholder="Nhập truy vấn tìm kiếm..."
              className="transition-all hover:drop-shadow-[0px_4px_10px_rgba(255,255,255,0.2)] placeholder:text-slate-400 text-white text-lg relative w-full h-12 indent-4 rounded-full bg-slate-900/50 border border-slate-700 focus:outline-none focus:ring-2 focus:ring-orange-500 shadow-inner"
              onClick={(e) => e.stopPropagation()}
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                // handleTranslate(e.target.value);
                document.getElementById("translate").style.display = "block";
              }}
              onFocus={(e) => {
                document.getElementById("translate").style.display = "block";
              }}
            ></input>
            <div className=" mr-2 w-20 h-10 gap-1 flex space-around items-center">
              <SpeechToText setQuery={setQuery} />
              <button
                type="button"
                id="mainsearch_button"
                className="border-orange-400 focus:bg-gradient-to-tr hover:opacity-100 border hover:bg-gradient-to-tr from-orange-400 via-red-500 to-red-400 duration-75  hover:scale-90  p-1 bg-slate-100 rounded-full"
                onClick={() => {
                  getImgLinks();
                  getRec();
                }}
              >
                <AiOutlineSearch color={"white"} fontSize="1.8rem" />
              </button>
            </div>
          </div>
          <div className="checkboxes flex flex-wrap items-center mt-3 h-fit gap-2 w-full justify-center">
            <Tabs
              queryHistory={queryHistory}
              handleHistory={handleHistory}
              selected={selected}
              setSelected={setSelected}
            />
            <button
              onClick={() => {
                clearAll();
              }}
              type="button"
              className="w-10 h-8 rounded-md bg-slate-500 hover:bg-orange-600 hover:ring-2 ring-orange-400 transition hover:scale-90"
            >
              Clear
            </button>
            <Select selected={selectedFilter} setSelected={setSelectedFilter} />
            <input
              tabIndex={-1}
              id="rangeFilter"
              type="number"
              placeholder="range"
              className="w-6 appearance-none transition-all hover:drop-shadow-[0px_4px_3px_rgba(255,255,255,0.2)] placeholder:italic text-slate-300 text-lg relative p-0.5 rounded-md bg-slate-800"
              onChange={(e) => {
                const val = e.target.value === "" ? 3 : Number(e.target.value);
                SetRangeFilter(Number.isNaN(val) ? 3 : val);
              }}
              value={rangeFilter}
            ></input>
            <div id="filter" className="flex items-center ">
              <input
                checked={filter}
                onChange={(e) => {
                  setFilter(e.target.checked);
                }}
                disabled={queryHistory.length === 0 && filter === false}
                id="Filter"
                type="checkbox"
                className="cursor-pointer rounded-md duration-200 w-5 h-5 accent-slate-600 bg-gray-100 border-gray-300 rounded hover:ring-slate-500 hover:ring-2"
              />
              <label
                htmlFor="Filter"
                className="cursor-pointer pl-0.5 text-slate-300"
              >
                <span className="">Filter</span>
              </label>
            </div>
            <div id="ignore" className="flex items-center ">
              <input
                checked={ignore}
                onChange={(e) => {
                  setIgnore(e.target.checked);
                }}
                id="Ignore"
                type="checkbox"
                className="cursor-pointer rounded-md duration-200 w-5 h-5 accent-slate-600 bg-gray-100 border-gray-300 rounded hover:ring-slate-500 hover:ring-2"
              />
              <label
                htmlFor="Ignore"
                className="cursor-pointer pl-0.5 text-slate-300"
              >
                <span className="">Ignore</span>
              </label>
            </div>
            <div id="Auto" className="flex items-center">
              <input
                checked={autoIgnore}
                onChange={(e) => {
                  setAutoIgnore(e.target.checked);
                }}
                id="AutoIgnore"
                type="checkbox"
                className="cursor-pointer rounded-md duration-200 w-5 h-5 accent-slate-600 bg-gray-100 border-gray-300 rounded hover:ring-slate-500 hover:ring-2"
              />
              <label
                htmlFor="AutoIgnore"
                className="cursor-pointer pl-0.5 text-slate-300"
              >
                <span className="">AutoIgnore</span>
              </label>
            </div>
            {/* question dropdown */}
            <div className="h-fit w-fit flex flex-col relative">
              <input
                placeholder="Questions"
                id="questionName"
                value={questionName}
                onKeyDown={(e) => {
                  if (e.key == "Enter") {
                    document.getElementById("send").click();
                  }
                }}
                onChange={(e) => {
                  setQuestionName(e.target.value);
                }}
                onClick={(e) => {
                  e.stopPropagation();
                  getOwnedQuestions(username);
                  document.getElementById("questions").style.display = "flex";
                  console.log("click");
                }}
                onFocus={() => {
                  document.getElementById("questions").style.display = "flex";
                }}
                className="transition-all hover:drop-shadow-[0px_2px_1px_rgba(255,255,255,0.2)] placeholder:italic text-slate-300 text-sm relative w-24 p-1 indent-1 rounded-full bg-slate-800"
              />
              <Questions
                isLoading={questionsLoading}
                questions={questions}
                username={username}
                setQuestionName={setQuestionName}
              />
            </div>
            <QueryKind
              meta={searchMeta}
              alignMode={alignMode}
              setAlignMode={setAlignMode}
            />
            <ChannelModes
              modes={channelModes}
              setModes={setChannelModes}
              meta={searchMeta}
            />
            <button
              type="button"
              disabled={autofilling}
              title="Lấp đủ 100 dòng đáp án bằng MMR trên kết quả đang hiển thị. Các dòng bạn tự chọn được giữ nguyên ở đầu."
              onClick={handleAutofill}
              className="text-xs px-1.5 h-6 rounded-md bg-slate-700 hover:bg-amber-700 hover:ring-1 ring-amber-400 transition text-slate-300 disabled:opacity-50"
            >
              {autofilling ? "..." : "Lấp 100"}
            </button>
            <button
              type="button"
              title="Export CSV cho câu hỏi đang chọn"
              onClick={() => {
                if (!questionName) {
                  alert("Chọn câu hỏi trước khi export CSV");
                  return;
                }
                window.open(
                  `${socket_url}/export/kis?questionName=${encodeURIComponent(questionName)}`,
                  "_blank"
                );
              }}
              className="text-xs px-1.5 h-6 rounded-md bg-slate-700 hover:bg-emerald-700 hover:ring-1 ring-emerald-400 transition text-slate-300"
            >
              CSV
            </button>
            <button
              type="button"
              title="Export toàn bộ submission ZIP"
              onClick={() => {
                window.open(`${socket_url}/export/submission_zip`, "_blank");
              }}
              className="text-xs px-1.5 h-6 rounded-md bg-slate-700 hover:bg-blue-700 hover:ring-1 ring-blue-400 transition text-slate-300"
            >
              ZIP
            </button>
            {/* <input
              placeholder="Username"
              value={username}
              onChange={(e) => handleUsername(e.target.value)}
              className="transition-all w-14 hover:drop-shadow-[0px_2px_1px_rgba(255,255,255,0.2)] placeholder:italic text-slate-300 text-sm relative  p-1 indent-1 rounded-full bg-slate-800"
            /> */}
            <div className="relative">
              <input
                tabIndex={1}
                id="username"
                value={username}
                autoComplete="off"
                // disabled={lockUsernameInput}
                readOnly={lockUsernameInput}
                onKeyDown={(e) => {
                  if (e.key == "Enter") {
                    document.getElementById("lock").click();
                  }
                }}
                type="search"
                placeholder="Username..."
                className={`w-24 transition-all  
              placeholder:italic text-slate-300 relative indent-1 rounded-md 
              ${
                lockUsernameInput
                  ? "bg-transparent border border-white/50 outline-none cursor-no-drop"
                  : "bg-slate-800"
              }
              `}
                onChange={(e) => {
                  handleUsername(e.target.value);
                }}
              />
              <Lock lock={lockUsernameInput} setLock={setLockUsernameInput} />
            </div>
            <input
              tabIndex={-1}
              id="search space"
              min={0}
              max={5}
              type="number"
              placeholder="Space"
              className="w-6 appearance-none transition-all hover:drop-shadow-[0px_4px_3px_rgba(255,255,255,0.2)] placeholder:italic text-slate-300 text-lg relative p-0.5 rounded-md bg-slate-800"
              onChange={(e) => {
                const val = e.target.value === "" ? 0 : Number(e.target.value);
                setSearchSpace(Number.isNaN(val) ? 0 : val);
              }}
              value={searchSpace}
            ></input>
            <button
              type="button"
              className="text-center items-center h-fit w-fit rounded-md bg-slate-500 hover:bg-orange-600 transition"
            >
              <a href="/submit" target="_blank" className="h-8 w-12 rounded-md">
                View
              </a>
            </button>

            <div className="flex items-center ml-auto feedbackmode text-orange-500 rounded-md">
              <input
                checked={feedbackMode}
                onChange={(e) => {
                  deleteFeedback();
                  setFeedbackMode(e.target.checked);
                }}
                id="Feedback"
                type="checkbox"
                className="cursor-pointer rounded-md  duration-200 w-5 h-5 accent-orange-700/75 rounded hover:ring-orange-300 hover:ring-2"
              />
              <label
                htmlFor="Feedback"
                className="cursor-pointer pl-0.5 text-slate-300"
              >
                <span className=" text-green-500">Feedback</span>
              </label>
            </div>
            <button
              id="send"
              onClick={() => {
                sendFeedback();
              }}
              type="button"
              className="text-center items-center px-1 h-8 mr-2 rounded-md bg-slate-500 hover:bg-orange-600 hover:ring-2 ring-orange-400 transition hover:scale-90"
            >
              Send
            </button>
          </div>
        </div>

        {/* {images} */}
        <div
          id="images"
          className=" flex-auto flex-col overflow-auto flex h-full"
        >
          {!loading &&
            videos.length > 0 &&
            videos
              .slice(
                page * VIDEO_PER_PAGE,
                page * VIDEO_PER_PAGE + VIDEO_PER_PAGE
              )
              .map((video, indexVideo) => {
                const video_info = video.video_info;
                // const currentVideos = (
                //   <VideoWrapper
                //     id={video.video_id}
                //     handleIgnore={() => handleIgnore(video_info.lst_idxs)}
                //   >
                //     {video_info.lst_keyframe_paths.map((path, index) => {
                //       let id = video_info.lst_idxs[index];
                //       return (
                //         <ImageListVideo
                //           imagepath={path}
                //           id={id}
                //           id_show={video_info.lst_keyframe_idxs[index]}
                //           handleKNN={handleKNN}
                //           handleSelect={handleSelect}
                //           feedbackMode={feedbackMode}
                //           handleFeedback={handleFeedback}
                //           handleIgnore={handleIgnore}
                //           imgFeedback={getImgFeedback(id)}
                //           toggleFullScreen={() =>
                //             toggleFullScreen({
                //               imgpath: path,
                //               id: id,
                //             })
                //           }
                //         />
                //       );
                //     })}
                //   </VideoWrapper>
                // );
                return "video_info_prev" in video ? (
                  <>
                    <VideoWrapper
                      filterFB={true}
                      id={video.video_id}
                      handleIgnore={() => handleIgnore(video_info.lst_idxs)}
                    >
                      {video_info.lst_keyframe_paths.map((path, index) => {
                        let id = video_info.lst_idxs[index];
                        return (
                          <ImageListVideo
                            addView={addView}
                            imagepath={path}
                            questionName={questionName}
                            id={id}
                            id_show={video_info.lst_keyframe_idxs[index]}
                            handleKNN={handleKNN}
                            handleSelect={() =>
                              handleSelect(
                                video_info.lst_keyframe_idxs[index],
                                video.video_id
                              )
                            }
                            feedbackMode={feedbackMode}
                            handleFeedback={handleFeedback}
                            handleIgnore={handleIgnore}
                            imgFeedback={getImgFeedback(id)}
                            isIgnored={getIgnoredImages(id)}
                            toggleFullScreen={() =>
                              toggleFullScreen({ imgpath: path, id: id })
                            }
                          />
                        );
                      })}
                    </VideoWrapper>
                    <VideoWrapper
                      filterFB={true}
                      id={`${video.video_id} PREV`}
                      handleIgnore={() =>
                        handleIgnore(video.video_info_prev.lst_idxs)
                      }
                      // isIgnored={getIsIgnored(indexVideo)}
                    >
                      {video.video_info_prev.lst_keyframe_paths.map(
                        (path, index) => {
                          let id = video.video_info_prev.lst_idxs[index];
                          return (
                            <ImageListVideo
                              addView={addView}
                              imagepath={path}
                              questionName={questionName}
                              id={id}
                              id_show={
                                video.video_info_prev.lst_keyframe_idxs[index]
                              }
                              handleKNN={handleKNN}
                              feedbackMode={false}
                              handleFeedback={handleFeedback}
                              handleSelect={() =>
                                handleSelect(
                                  video.video_info_prev.lst_keyframe_idxs[
                                    index
                                  ],
                                  video.video_id
                                )
                              }
                              isIgnored={getIgnoredImages(id)}
                              handleIgnore={handleIgnore}
                              imgFeedback={""}
                              toggleFullScreen={() =>
                                toggleFullScreen({
                                  imgpath: path,
                                  id: id,
                                })
                              }
                            />
                          );
                        }
                      )}
                    </VideoWrapper>
                    <hr class="h-2 border-1 my-8 bg-orange-400 border-slate-700"></hr>
                  </>
                ) : (
                  <>
                    <ExplainBadge
                      explain={video.explain}
                      why={video.why}
                      rrfScore={video.rrf_score}
                    />
                    <VideoWrapper
                      id={video.video_id}
                      handleIgnore={() => handleIgnore(video_info.lst_idxs)}
                    >
                      {video_info.lst_keyframe_paths.map((path, index) => {
                        let id = video_info.lst_idxs[index];
                        return (
                          <ImageListVideo
                            addView={addView}
                            imagepath={path}
                            questionName={questionName}
                            id={id}
                            id_show={video_info.lst_keyframe_idxs[index]}
                            handleKNN={handleKNN}
                            handleSelect={() =>
                              handleSelect(
                                video_info.lst_keyframe_idxs[index],
                                video.video_id
                              )
                            }
                            feedbackMode={feedbackMode}
                            handleFeedback={handleFeedback}
                            isIgnored={getIgnoredImages(id)}
                            handleIgnore={handleIgnore}
                            imgFeedback={getImgFeedback(id)}
                            toggleFullScreen={() =>
                              toggleFullScreen({
                                imgpath: path,
                                id: id,
                              })
                            }
                          />
                        );
                      })}
                    </VideoWrapper>
                    <hr class="border-1 my-6 bg-orange-400 border-slate-700"></hr>
                  </>
                );
              })}
        </div>
        {/* buttons */}
        {videos.length > 0 && !loading && (
          <PageButton
            totalPage={Math.floor(videos.length / 7)}
            autoFetch={autoFetch}
            isFilter={checkFilter()}
            showAutoFetch={showAutoFetch}
            page={page}
            setPage={setPage}
            autoIgnore={autoIgnore}
            handleAutoIgnore={handleAutoIgnore}
            DivID={"images"}
          />
        )}
      </div>
    </div>
  );
}

export default Index;
