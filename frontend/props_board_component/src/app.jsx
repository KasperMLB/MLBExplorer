import React from "react";
import { PropsBoard } from "./props_board.jsx";
import { ZoneTool } from "./zone_tool.jsx";

export function App({ args }) {
  if (args?.componentType === "zone_tool") {
    return <ZoneTool args={args} />;
  }
  return <PropsBoard args={args} />;
}
