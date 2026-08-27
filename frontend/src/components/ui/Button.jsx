import React from "react";

// Nút. Một chỗ duy nhất quyết định nút trông thế nào.
//
// variant: "default" | "accent" | "good" | "danger"
// size:    "md" | "sm" | "icon"

function Button({
  variant = "default",
  size = "md",
  active = false,
  className = "",
  children,
  ...rest
}) {
  const cls = [
    "btn",
    variant !== "default" ? `btn--${variant}` : "",
    size !== "md" ? `btn--${size}` : "",
    active ? "btn--on" : "",
    className,
  ]
    .filter(Boolean)
    .join(" ");
  return (
    <button type="button" className={cls} {...rest}>
      {children}
    </button>
  );
}

export default Button;
