import React from "react";
import ReactDOM from "react-dom/client";
import { Streamlit, withStreamlitConnection } from "streamlit-component-lib";
import { PropsBoard } from "./props_board.jsx";

const ConnectedBoard = withStreamlitConnection(PropsBoard);

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ConnectedBoard />
  </React.StrictMode>,
);

Streamlit.setFrameHeight(760);
