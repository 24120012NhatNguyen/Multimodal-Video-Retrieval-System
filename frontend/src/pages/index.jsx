import { useEffect, useRef, useState } from "react";
import React from "react";
import { AiOutlineSearch } from "react-icons/ai";
import Select, { FILTER_OPTIONS } from "../components/Select.jsx";
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
import TrakePanel from "../components/TrakePanel.jsx";
import Button from "../components/ui/Button.jsx";
import Segmented from "../components/ui/Segmented.jsx";
import Field from "../components/ui/Field.jsx";
import Group from "../components/ui/Group.jsx";
import Check from "../components/ui/Check.jsx";
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
  // [] chứ KHÔNG phải {}: mọi chỗ dưới đây đọc videos.length và videos.slice().
  // Khởi tạo bằng object thì videos.length là undefined, nên nhánh "chưa có kết
  // quả" không bao giờ chạy và vùng kết quả chỉ là một khoảng đen.
  const [videos, setVideos] = useState([]);
  const [id, setId] = useState([]);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState(false);
  const [loading, setLoading] = useState(false);
  const [recTags, setRecTags] = useState([]);
  const [fullScreenImg, setFullScreenImg] = useState(null);
  const [queryHistory, setQueryHistory] = useState([]);
  const [k, setK] = useState(500);
  const [selected, setSelected] = useState(queryHistory[0]);
  const [selectedFilter, setSelectedFilter] = useState(FILTER_OPTIONS[0]);
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
  // Phân rã truy vấn bằng LLM. MẶC ĐỊNH TẮT — đo được là làm giảm điểm
  // (0.60 -> 0.46 theo công thức chấm của BTC). Vẫn để người dùng bật khi gặp
  // truy vấn khó mà dịch máy dịch sai.
  const [useLlm, setUseLlm] = useState(false);
  // Bảng object là công cụ PHỤ nhưng đang chiếm 728px cố định — gần 40% màn
  // hình 1920. Thu gọn được, và mặc định gọn.
  const [panelOpen, setPanelOpen] = useState(false);
  // TRAKE là dạng bài riêng: chuỗi khoảnh khắc theo thứ tự, nộp một dãy frame.
  // Nó không dùng chung lưới kết quả với KIS nên có khung riêng.
  const [trakeOpen, setTrakeOpen] = useState(false);
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
      // Không dùng alert(): hộp thoại chặn cả luồng dựng trang, và người dùng
      // gặp nó mỗi lần mở tab mới. Mở khoá ô tên + focus là đủ nói.
      setLockUsernameInput(false);
      setTimeout(() => {
        const el = document.getElementById("username");
        if (el) el.focus();
      }, 0);
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
    // So theo `value`, không theo nhãn hiển thị.
    let filtervideo = selectedFilter.value ?? 0;
    fetch(`${web_url}/textsearch`, {
      method: "post",
      headers: apiHeaders(),
      body: JSON.stringify({
        // Che do hop nhat cap video: xep hang video bang nhieu kenh bang chung
        // roi moi xuong frame. Duong cu (xep hang frame truc tiep) chay tren
        // dict/ da bi xoa nen khong con dung duoc.
        fusion: true,
        decompose: useLlm,
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
    setSelectedFilter(FILTER_OPTIONS[0]);
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
    // encodeURIComponent BAT BUOC: khoa keyframe la "L30_V046#4865" va dau '#'
    // trong URL la ky tu mo FRAGMENT -- trinh duyet cat tu do tro di, backend chi
    // nhan duoc "L30_V046" roi tra 404 "khong tim thay keyframe".
    fetch(`${web_url}/imgsearch?imgid=${encodeURIComponent(imgId)}&k=${k}`, fetchGetObj)
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
      fetch(`${web_url}/relatedimg?imgid=${encodeURIComponent(image.id)}`, fetchGetObj)
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

  // Hỏi VLM trên một frame. Chạy ở SERVER LOCAL: /qa cần ảnh thật, mà ảnh chỉ
  // trích được từ data/videos ở máy này.
  const askVlm = async (entryKey) => {
    const at = String(entryKey).lastIndexOf("#");
    if (at < 0) {
      alert("Khoá keyframe không hợp lệ: " + entryKey);
      return "";
    }
    if (!query.trim()) {
      alert("Gõ câu hỏi vào ô tìm kiếm trước đã.");
      return "";
    }
    try {
      const res = await fetch(`${socket_url}/qa`, {
        method: "post",
        headers: apiHeaders(),
        body: JSON.stringify({
          video_id: String(entryKey).slice(0, at),
          frame_idx: parseInt(String(entryKey).slice(at + 1), 10),
          question: query,
          window_sec: 3,
          n_frames: 3,
        }),
      });
      const d = await res.json();
      if (d.error) {
        alert("Hỏi VLM lỗi: " + d.error);
        return "";
      }
      if (d.degraded) {
        alert(
          "VLM không dùng được — đây chỉ là text OCR/ASR đọc được quanh frame:\n" +
            (d.answer || "(trống)")
        );
      }
      return d.answer || "";
    } catch (e) {
      alert("Không gọi được /qa ở " + socket_url + ": " + e);
      return "";
    }
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

          let filtervideo = selectedFilter.value ?? 0;
          fetch(`${web_url}/textsearch`, {
            method: "post",
            headers: apiHeaders(),
            body: JSON.stringify({
              fusion: true,
              decompose: useLlm,
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
            // Gửi kèm truy vấn để backend lấy thêm ứng viên NGOÀI lưới đang
            // hiện. Bài nộp có 100 ô chấm theo thứ hạng — không có lý do gì để
            // 100 ô đó bị giới hạn bởi số ảnh người dùng đang nhìn thấy.
            query_vi: query,
            kind: searchMeta && searchMeta.query ? searchMeta.query.kind : null,
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
      {panelOpen && (
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
      )}
      <div className="relative flex-auto h-full flex flex-col overflow-hidden min-w-0">
        {/* {loading icon} */}
        {loading && <LoadingIcon />}
        {/* {searchbars} */}
        {/* ================= THANH CÔNG CỤ =================================
            Xếp theo VIỆC, không phải theo thứ tự lịch sử code:
              hàng 1  tìm gì          — ô tìm kiếm là thứ to nhất màn hình
              hàng 2  tìm thế nào     — các công tắc điều chỉnh cách tìm
              hàng 3  làm gì với đáp án — chọn câu hỏi, lấp 100, xuất bài
            ================================================================= */}
        <div className="appbar w-full sticky top-0 z-20 px-4 py-3 flex flex-col gap-2.5">

          {/* ---------- hàng 1: tìm gì ---------- */}
          <div id="bar" className="w-full flex relative gap-2 items-center">
            <span
              id="translate"
              onClick={(e) => {
                e.stopPropagation();
                navigator.clipboard.writeText(translate);
                document.getElementById("mainsearch").focus();
              }}
              style={{ zIndex: 2, display: "none" }}
              title="Bản dịch gửi cho SigLIP. Bấm để chép."
              className="panel absolute top-[52px] left-0 right-0 cursor-pointer px-3 py-2 text-sm text-[color:var(--ink-2)] hover:border-[color:var(--accent)] transition"
            >
              {translate ? translate : "Bản dịch sẽ hiện ở đây..."}
            </span>

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
              placeholder="Mô tả cảnh cần tìm — vd: ba người đi bộ xuống dốc dưới mưa, hai người cầm dù"
              className="searchbar"
              onClick={(e) => e.stopPropagation()}
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                document.getElementById("translate").style.display = "block";
              }}
              onFocus={() => {
                document.getElementById("translate").style.display = "block";
              }}
            />

            <SpeechToText setQuery={setQuery} />

            <Button
              id="mainsearch_button"
              variant="accent"
              className="h-[46px] px-5 text-[15px]"
              onClick={() => {
                getImgLinks();
                getRec();
              }}
            >
              <AiOutlineSearch fontSize="1.15rem" />
              Tìm
            </Button>

            <Field
              label="Số ảnh"
              hint="Số keyframe hiển thị trên lưới. Bài nộp không bị giới hạn bởi con số này."
              id="K"
              type="number"
              inputClassName="inp--num h-[46px]"
              onChange={(e) => {
                const val = e.target.value === "" ? 500 : Number(e.target.value);
                const parsed = Number.isNaN(val) ? 500 : val;
                if (filter && parsed > currentK)
                  alert("Chế độ Lọc: số ảnh phải nhỏ hơn lượt tìm trước");
                else setK(parsed);
              }}
              value={k}
            />
          </div>

          {/* ---------- hàng 2: tìm thế nào ---------- */}
          <div className="w-full flex flex-wrap items-center gap-2">
            <ChannelModes
              modes={channelModes}
              setModes={setChannelModes}
              meta={searchMeta}
            />

            <QueryKind
              meta={searchMeta}
              alignMode={alignMode}
              setAlignMode={setAlignMode}
            />

            <Group label="LLM">
              <Segmented
                value={useLlm ? "on" : "off"}
                onChange={(v) => setUseLlm(v === "on")}
                options={[
                  { value: "off", label: "Tắt",
                    title: "Mặc định. Dùng dịch máy — đo được là cho điểm cao hơn." },
                  { value: "on", label: "Bật", tone: "on",
                    title: "Bật khi dịch máy dịch sai thuật ngữ Việt (vd 'múa lân' ra 'the unicorn'). Đo được: bật vào làm điểm giảm 0.60 → 0.46, nên chỉ bật cho truy vấn khó." },
                ]}
              />
            </Group>

            <Group label="Lọc lại">
              <Check
                id="Filter"
                checked={filter}
                disabled={queryHistory.length === 0 && filter === false}
                onChange={(e) => setFilter(e.target.checked)}
                hint="Chỉ tìm trong những video đã ra ở lượt trước. Bật cái này thì ô Hướng bên cạnh mới có tác dụng."
              >
                Trong kết quả cũ
              </Check>
              <Check
                id="Ignore"
                checked={ignore}
                onChange={(e) => setIgnore(e.target.checked)}
                hint="Bỏ qua các frame đã đánh dấu loại"
              >
                Bỏ frame đã loại
              </Check>
              <Check
                id="AutoIgnore"
                checked={autoIgnore}
                onChange={(e) => setAutoIgnore(e.target.checked)}
                hint="Tự đánh dấu loại các frame đã xem qua"
              >
                Tự loại đã xem
              </Check>
            </Group>

            {/* "Dải" và "Nhóm" đã bị gỡ: backend chưa bao giờ đọc hai tham số
                đó, nên chúng là nút chết — người dùng chỉnh số rồi tưởng có tác
                dụng. Thà không có còn hơn có mà im lặng không làm gì. */}
            <Group label="Hướng">
              <Select selected={selectedFilter} setSelected={setSelectedFilter} />
            </Group>

            <Button
              size="sm"
              variant="danger"
              onClick={() => clearAll()}
              title="Xoá kết quả và mọi bộ lọc đang bật"
            >
              Xoá hết
            </Button>

            <div className="ml-auto flex items-center gap-2">
              <Tabs
                queryHistory={queryHistory}
                handleHistory={handleHistory}
                selected={selected}
                setSelected={setSelected}
              />
              <Button
                size="sm"
                active={panelOpen}
                onClick={() => setPanelOpen((v) => !v)}
                title="Tìm theo lớp object và VỊ TRÍ trên khung hình, kèm ô lọc theo chữ trên hình / lời nói"
              >
                {panelOpen ? "Ẩn bảng object" : "Bảng object"}
              </Button>
              <Button
                size="sm"
                active={trakeOpen}
                onClick={() => setTrakeOpen((v) => !v)}
                title="Dạng bài TRAKE: tìm một CHUỖI khoảnh khắc theo thứ tự trong cùng một video, rồi nộp cả dãy frame"
              >
                {trakeOpen ? "Ẩn TRAKE" : "TRAKE"}
              </Button>
            </div>
          </div>

          {/* ---------- hàng 3: làm gì với đáp án ---------- */}
          <div className="w-full flex flex-wrap items-center gap-2">
            <Group label="Câu hỏi">
              <div className="h-fit w-fit flex flex-col relative">
                <input
                  placeholder="Chọn hoặc gõ tên..."
                  id="questionName"
                  value={questionName}
                  onKeyDown={(e) => {
                    if (e.key == "Enter") document.getElementById("send").click();
                  }}
                  onChange={(e) => setQuestionName(e.target.value)}
                  onClick={(e) => {
                    e.stopPropagation();
                    getOwnedQuestions(username);
                    document.getElementById("questions").style.display = "flex";
                  }}
                  onFocus={() => {
                    document.getElementById("questions").style.display = "flex";
                  }}
                  className="inp inp--sm w-40"
                />
                <Questions
                  isLoading={questionsLoading}
                  questions={questions}
                  username={username}
                  setQuestionName={setQuestionName}
                />
              </div>
            </Group>

            <Group label="Bài nộp">
              <Button
                size="sm"
                variant="accent"
                disabled={autofilling}
                title="Lấp đủ 100 dòng đáp án. Ô 1–20 giữ nguyên thứ hạng tìm kiếm, ô 21–100 mới trải đều theo thời gian. Frame bạn tự chọn luôn nằm đầu."
                onClick={handleAutofill}
              >
                {autofilling ? "Đang lấp..." : "Lấp 100 dòng"}
              </Button>
              <Button
                size="sm"
                title="Tải CSV của câu hỏi đang chọn"
                onClick={() => {
                  if (!questionName) {
                    alert("Chọn câu hỏi trước khi tải CSV");
                    return;
                  }
                  window.open(
                    `${socket_url}/export/kis?questionName=${encodeURIComponent(questionName)}`,
                    "_blank"
                  );
                }}
              >
                Tải CSV
              </Button>
              <Button
                size="sm"
                title="Tải toàn bộ bài nộp dạng ZIP"
                onClick={() => window.open(`${socket_url}/export/submission_zip`, "_blank")}
              >
                Tải ZIP
              </Button>
              <Button
                size="sm"
                title="Mở trang xem lại đáp án đã chọn"
                onClick={() => window.open("/submit", "_blank")}
              >
                Xem đáp án
              </Button>
            </Group>

            <Group label="Phản hồi">
              <Check
                id="Feedback"
                checked={feedbackMode}
                onChange={(e) => {
                  deleteFeedback();
                  setFeedbackMode(e.target.checked);
                }}
                hint="Bật để đánh dấu ảnh đúng/sai rồi gửi lại cho hệ tìm kiếm"
              >
                Chấm ảnh
              </Check>
              <Button
                id="send"
                size="sm"
                disabled={!feedbackMode}
                onClick={() => sendFeedback()}
                title="Gửi các ảnh đã chấm để tìm lại"
              >
                Gửi &amp; tìm lại
              </Button>
            </Group>

            <div className="ml-auto flex items-center gap-2">
              <div className="relative">
                <input
                  tabIndex={1}
                  id="username"
                  value={username}
                  autoComplete="off"
                  readOnly={lockUsernameInput}
                  onKeyDown={(e) => {
                    if (e.key == "Enter") document.getElementById("lock").click();
                  }}
                  type="search"
                  placeholder="Tên của bạn..."
                  className={`inp inp--sm w-36 pr-7 ${
                    lockUsernameInput ? "cursor-not-allowed opacity-70" : ""
                  }`}
                  onChange={(e) => handleUsername(e.target.value)}
                />
                <Lock lock={lockUsernameInput} setLock={setLockUsernameInput} />
              </div>
            </div>
          </div>
        </div>

        {/* {images} */}
        {trakeOpen && (
          <div className="flex-auto overflow-auto border-t border-[color:var(--line)]">
            <TrakePanel
              questionName={questionName}
              addView={addView}
              onClose={() => setTrakeOpen(false)}
            />
          </div>
        )}

        {/* ---------- thanh trạng thái kết quả ---------- */}
        {!trakeOpen && !loading && searchMeta && (
          <div className="flex flex-wrap items-center gap-2 px-4 py-2 text-[12px]
                          text-[color:var(--ink-2)] border-b border-[color:var(--line)]">
            <span className="mono">
              {videos.length} video · {id.length} khung hình
            </span>
            {searchMeta.nRanked > 0 && (
              <span className="text-[color:var(--muted)]">
                (xếp hạng trên {searchMeta.nRanked} video)
              </span>
            )}
            {searchMeta.query && searchMeta.query.query_en && (
              <span className="chip" title="Chuỗi thật sự gửi cho SigLIP">
                {searchMeta.query.query_en.split("\n")[0].slice(0, 70)}
              </span>
            )}
            {Object.entries(searchMeta.errors || {}).map(([k, v]) => (
              <span key={k} className="chip chip--warn" title={String(v)}>
                kênh {k} lỗi
              </span>
            ))}
          </div>
        )}

        <div
          id="images"
          className={`results flex-auto flex-col overflow-auto flex h-full px-3 pb-2 ${
            trakeOpen ? "hidden" : ""
          }`}
        >
          {/* ---------- trạng thái rỗng ---------- */}
          {!loading && videos.length === 0 && (
            <div className="flex flex-col items-center justify-center gap-3 h-full text-center px-6">
              <div className="text-[15px] text-[color:var(--ink-2)]">
                {searchMeta
                  ? "Không có kết quả nào khớp."
                  : "Mô tả cảnh cần tìm rồi bấm Tìm."}
              </div>
              <div className="text-[12.5px] text-[color:var(--muted)] max-w-[54ch] leading-relaxed">
                {searchMeta ? (
                  <>
                    Thử tắt bớt bộ lọc, hoặc bật kênh <b>Chữ trên hình</b> / <b>Lời nói</b>
                    {" "}nếu truy vấn nhắc tới chữ hoặc tên riêng.
                  </>
                ) : (
                  <>
                    Mô tả càng cụ thể càng tốt — màu sắc, số lượng, vật thể lạ.
                    Cần tìm theo vị trí trên khung hình thì dùng bảng object bên trái.
                  </>
                )}
              </div>
            </div>
          )}

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
                            askVlm={askVlm}
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
                              askVlm={askVlm}
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
                    <hr className="my-6 h-px border-0 bg-[color:var(--accent-dim)]" />
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
                            askVlm={askVlm}
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
